// Extraction shim around M3S, the multiscale sparse Sinkhorn solver of Chen,
// Li and Lu (CVPR 2022), upstream ucascnic/CudaOT at the SHA in ../PIN.
//
// Upstream's API image_ot_solver(image_size, muX, muY) returns the OT distance
// only. This file #includes upstream's SparseSinkhorn/solver.cu (include-guarded,
// and #included the same way by upstream's image_ot_solver.cu) and replicates the
// cuda_image_ot_solver<image_size> body (solver.cu:1241-1631). The algorithm, its
// ε ladder and its hard-wired parameters run as published, and no upstream file
// is edited. The first-party additions are the blob I/O, the end-of-solve copies
// of the plan and the duals (in place of upstream's output.csv dump into the
// working directory), per-layer iteration counters, timing fences, and early
// returns that record a nonzero msg.
//
// Plan. The delivered coupling is π_k = val[k] · u[row[k]] · v[col[k]], the
// expression upstream's record_cost and scoreTransportCost score the plan with
// (solver.cu:52, 747). The final absorption in solveSingle_ moves the scalings
// into alphaH, betaH and resets u = v = 1, so π_k = val[k].
//
// Duals. alpha.bin and beta.bin hold the finest-layer absorbed potentials of the
// stabilized scaling, whose kernel carries the reference measure:
//     K_ij = exp((α_i + β_j − c_ij)/ε) · μX_i · μY_j,   π = diag(u) K diag(v).
// They satisfy α_i + β_j − c_ij = ε log π_ij only after the measure is folded in,
// α_i += ε log μX_i and β_j += ε log μY_j with ε = eps_final from meta.json;
// src/runners/run_m3s.py does this.
//
// Costs are |x − y|² in integer pixel coordinates (p_index = 2); the runner
// divides them by 2(n_g − 1)² to get the max-1 cost.
//
// n_g = 1024 is refused. The size macros accept it, but the hierarchy always
// appends the 1024² original atoms as its finest layer, which overflows the
// MAXSIZE = 512 arrays XData::data[512²·3] (on the host, in create_dev_struct)
// and COO_Sparese_Matrix::crs_row_ptrl[512²+1] (on the device, in
// cusparseXcoo2csr).
//
// Exit codes: 0 ok; 2 when a shim-side CUDA call (SHIM_CHECK) reports
// cudaErrorMemoryAllocation; 3 nonzero upstream msg, written to meta.json;
// 1 usage, blob I/O or any other error. Upstream's CHECK macro exits 1 itself,
// printing the CUDA error string ("out of memory" on OOM) on stderr.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <cmath>

#include <SparseSinkhorn/solver.cu>

// The runner's cost rescale 1/(2(n_g − 1)²) assumes p = 2.
static_assert(p_index == 2,
              "m3s_shim: upstream p_index != 2; the sql2 cost-scale "
              "conversion (obj/(2(ng-1)^2)) would be wrong; re-verify.");

// solver.cu leaves its function-local MAXELEMENT macro defined for the rest of
// the translation unit; undefine it, or it would rewrite shim_solveSingle_'s
// template parameter MAXELEMENT.
#undef MAXELEMENT

// Upstream's CHECK exits 1 on every CUDA error; this one exits 2 on OOM.
#define SHIM_CHECK(call)                                                       \
{                                                                              \
    const cudaError_t err_ = (call);                                           \
    if (err_ != cudaSuccess) {                                                 \
        fprintf(stderr, "m3s_shim: CUDA error %s:%d code %d: %s\n",            \
                __FILE__, __LINE__, (int)err_, cudaGetErrorString(err_));      \
        exit(err_ == cudaErrorMemoryAllocation ? 2 : 1);                       \
    }                                                                          \
}

// Upstream solveSingle_ (solver.cu:181-367) plus one line: *iters_done grows by
// cfg->innerIterations after each iterate() call. Upstream's own nIterations
// skips batches that end in an absorption `continue` and is discarded on return.
// Otherwise identical, except for (long) casts in the three "kernel entries"
// printfs (upstream passes an int to %ld, which is undefined behaviour) and the
// omitted _RECORD_ block (_RECORD_ is never defined upstream). Upstream quirks
// are kept, e.g. familyY in the x-propagate calls (solver.cu:321, 355).
template< int ROW_,  int COL_, int MAXLAYER_, int MAXELEMENT>
int  shim_solveSingle_(int *msg_cu,int *kernelValid,
                 COO_Sparese_Matrix<MAXELEMENT> *kernel,
                 COO_Sparese_Matrix<MAXELEMENT> *kernelT,
                 CuSinkhornSolverParameters *cfg,
                 double eps,
                 double *muX_cuda,
                 double *muY_cuda,
                 Familys<ROW_*COL_,MAXLAYER_> *familyX,
                 Familys<ROW_*COL_,MAXLAYER_> *familyY,
                 int layerBottom,
                 XData * alphaH, XData *betaH,
                 XData *xpos, XData *ypos,
                 XData *rax,XData *ray,
                 int xres,int yres,
                 double *u,double *v,
                 double *res,int layer,
                 Ind_Ptr_ *indptrl_cu_mid,
                 Ind_Ptr_ *indptrl_cu,
                 COO_Buff * buff,
                 cublasHandle_t handle,
                 long long *iters_done /* first-party */)
{
    int nIterations=0;
    int nAbsorptionLoops=0;
    int msg = 0;
    int N ;

    double error = 0.0;
    cudaMemset(msg_cu,0,sizeof(int));

    if(!*kernelValid) {
         if(layer < layerBottom  ){
         // refine the whole region
           msg = refinekernel_2<MAXELEMENT>(kernel,kernelT,
                                  muX_cuda,
                                  muY_cuda,
                                  layerBottom,
                                  eps,xres,yres,
                                  layer,
                                  alphaH,betaH,xpos,ypos,rax,ray,
                                 indptrl_cu_mid,indptrl_cu,
                                   kernelValid,buff,cfg);
                                   }
        else
        {

            msg = refinekernel_3<MAXELEMENT>(kernel,kernelT,
                              muX_cuda,
                              muY_cuda,
                              layerBottom,
                              eps,xres,yres,
                              layer,
                              alphaH,betaH,xpos,ypos,rax,ray,
                             indptrl_cu_mid,indptrl_cu,
                               kernelValid,buff,cfg);

        }

         *kernelValid=1;
        if(msg!=0) return msg;
    }

    cudaMemcpy(&N,&kernel->nonZeros,sizeof(int),cudaMemcpyDeviceToHost);
    printf("\t\tkernel entries: %ld\n",(long)N);

    while(true) {


        iterate(msg_cu,cfg->innerIterations,
                         xres,muX_cuda,yres,muY_cuda,kernel,
                         kernelT,u,v,res
                         );
        *iters_done += cfg->innerIterations;   // first-party

        cudaMemcpy(&msg,msg_cu,sizeof(int),cudaMemcpyDeviceToHost);

        if(msg== MSG_NANSCALING ) return msg;
        checkAbsorb<<<xres/128+1,128>>>(msg_cu,u,v,cfg->absorption_scalingBound,xres,yres);
        cudaMemcpy(&msg,msg_cu,sizeof(int),cudaMemcpyDeviceToHost);

        if(msg == MSG_ABSORB_REITERATE) {

            cudaMemset(msg_cu,0,sizeof(int));
            printf("\t\tabsorbing\n");
            nAbsorptionLoops++;
            if(nAbsorptionLoops>cfg->maxAbsorptionLoops) {
                return MSG_ABSORB_TOOMANYABSORPTIONS;
            }
            SinkhornAbsorbScaling1<<<xres/128+1,128>>>(alphaH, u, layer, eps,xres);
            signal_propagate_doublex_<ROW_*COL_,MAXLAYER_><<<1,128>>>(familyX, alphaH, 0, layer, THierarchicalPartition::MODE_MAX);


            SinkhornAbsorbScaling2<<<yres/128+1,128>>>(betaH, v, layer, eps,yres);
            signal_propagate_doubley_<ROW_*COL_,MAXLAYER_><<<1,128>>>(familyY, betaH, 0, layer, THierarchicalPartition::MODE_MAX);

            msg = refinekernel_3<MAXELEMENT>(kernel,kernelT,
                              muX_cuda,
                              muY_cuda,
                              layerBottom,
                              eps,xres,yres,
                              layer,
                              alphaH,betaH,xpos,ypos,rax,ray,
                             indptrl_cu_mid,indptrl_cu,
                               kernelValid,buff,cfg);



            cudaMemcpy(&N,&kernel->nonZeros,sizeof(int),cudaMemcpyDeviceToHost);
            printf("\t\tkernel entries: %ld\n",(long)N);


            if(msg!=0) return msg;
            *kernelValid=1;
            continue;
        } else {
            if(msg!=0) return msg;
        }
        nAbsorptionLoops=0;

        error = getError(kernel,
                 muX_cuda,
                 u,v,res,xres,yres,handle);
        if(isnan(error)){
            return -2 ;
        }
        if(error<=cfg->maxError) {
            cudaMemset(msg_cu,0,sizeof(int));
            checkAbsorb<<<xres/128+1,128>>>(msg_cu,u,v,cfg->absorption_scalingLowerBound,xres,yres);
            cudaMemcpy(&msg,msg_cu,sizeof(int),cudaMemcpyDeviceToHost);

            if(msg==MSG_ABSORB_REITERATE) {
                printf("\t\tsafety absorption.\n");

                SinkhornAbsorbScaling1<<<xres/128+1,128>>>(alphaH, u, layer, eps,xres);
                signal_propagate_doublex_<ROW_*COL_,MAXLAYER_><<<1,128>>>(familyY, alphaH, 0, layer, THierarchicalPartition::MODE_MAX);
                SinkhornAbsorbScaling2<<<yres/128+1,128>>>(betaH, v, layer, eps,yres);
                signal_propagate_doubley_<ROW_*COL_,MAXLAYER_><<<1,128>>>(familyY, betaH, 0, layer, THierarchicalPartition::MODE_MAX);

                if(layer < layerBottom)
                    msg=generateKernel_<MAXELEMENT>(kernel,kernelT,cfg,eps,
                                       muX_cuda,muY_cuda,layer,
                                       alphaH,betaH,xpos,ypos,rax,ray,xres,yres,
                                       indptrl_cu_mid,indptrl_cu,buff);
                else{
                    msg = refinekernel_2<MAXELEMENT>(kernel,kernelT,
                                      muX_cuda,
                                      muY_cuda,
                                      layerBottom,
                                      eps,xres,yres,
                                      layer,
                                      alphaH,betaH,xpos,ypos,rax,ray,
                                     indptrl_cu_mid,indptrl_cu,
                                       kernelValid,buff,cfg);

                }

                cudaMemcpy(&N,&kernel->nonZeros,sizeof(int),cudaMemcpyDeviceToHost);
                printf("\t\tkernel entries: %ld\n",(long)N);
                if(msg!=0) return msg;
                *kernelValid=1;

                continue;

            } else {
                // otherwise, check for other errors
                if(msg!=0) return msg;
                printf("\t\tfinal absorption\n");
                SinkhornAbsorbScaling1<<<xres/128+1,128>>>(alphaH, u, layer, eps,xres);
                signal_propagate_doublex_<ROW_*COL_,MAXLAYER_><<<1,128>>>(familyY, alphaH, 0, layer, THierarchicalPartition::MODE_MAX);
                SinkhornAbsorbScaling2<<<yres/128+1,128>>>(betaH, v, layer, eps,yres);
                signal_propagate_doubley_<ROW_*COL_,MAXLAYER_><<<1,128>>>(familyY, betaH, 0, layer, THierarchicalPartition::MODE_MAX);
                *kernelValid=0;
                return 0;
            }
        }
        nIterations+=cfg->innerIterations;
        if(nIterations>cfg->maxIterations) {
            return MSG_EXCEEDMAXITERATIONS;
        }
    }
}

struct ShimResult {
    int msg = 0;                       // 0 ok; an upstream msg, or -10 for a blob-write failure
    double distance_native = 0.0;      // calculate_result, native pixel² units
    double eps_final = 0.0;            // ε in effect at delivery
    double solve_seconds = 0.0;        // setup + ladder + delivery rebuild (CUDA-synced)
    double extract_seconds = 0.0;      // COO/dual copies + native distance
    long long nnz = 0;
    int xres_final = 0;
    int n_layers = 0;
    std::vector<long long> iters_per_layer;   // indexed by layer; 0 below layerCoarsest
};

static bool write_blob(const std::string &path, const void *data, size_t bytes) {
    FILE *f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "m3s_shim: cannot open %s\n", path.c_str()); return false; }
    size_t w = fwrite(data, 1, bytes, f);
    fclose(f);
    if (w != bytes) { fprintf(stderr, "m3s_shim: short write to %s\n", path.c_str()); return false; }
    return true;
}

// The cuda_image_ot_solver body (see the file header). With outdir == nullptr
// (the warm-up) nothing is copied out or written.
template<int image_size>
int shim_solve(std::vector<double> &muXdat,
               std::vector<double> &muYdat,
               const std::string *outdir,
               ShimResult &out){

    constexpr int kMaxEl = ( 1 << (ALLOCALTE_MEMORY(image_size)) );   // upstream's MAXELEMENT (solver.cu:1246)

    // upstream: solver.cu:1249-1318
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("device %d: %s \n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    SHIM_CHECK(cudaDeviceSynchronize());           // first-party timing fence
    double t_solve0 = seconds();                   // handler_cuda_error.h helper

    std::vector<int> muXdim = {image_size, image_size};
    std::vector<int> muYdim = {image_size, image_size};

    TDoubleMatrix muX,muY;

    muX.data=muXdat.data();
    muX.dimensions=muXdim.data();
    muX.depth=2;
    muY.data = muYdat.data();
    muY.dimensions = muYdim.data();
    muY.depth=2;

    int layerFinest=(GETLOG2(image_size)); // what is the finest layer we want to solve?
    int layerCoarsest = 3; // coarsest layer to solve on. sometimes skip a few layers at the top
    int msg; // store return codes from functions

    TMultiScaleSetupSingleGrid MultiScaleSetupX(&muX,layerFinest);
    msg=MultiScaleSetupX.Setup();
    if(msg!=0) { printf("error: %d\n",msg); out.msg = msg; return msg; }
    msg=MultiScaleSetupX.SetupRadii();
    if(msg!=0) { printf("error: %d\n",msg); out.msg = msg; return msg; }
    msg=MultiScaleSetupX.SetupDuals();
    if(msg!=0) { printf("error: %d\n",msg); out.msg = msg; return msg; }


    TMultiScaleSetupSingleGrid MultiScaleSetupY(&muY,layerFinest);
    msg=MultiScaleSetupY.Setup();
    if(msg!=0) { printf("error: %d\n",msg); out.msg = msg; return msg; }
    msg=MultiScaleSetupY.SetupRadii();
    if(msg!=0) { printf("error: %d\n",msg); out.msg = msg; return msg; }
    msg=MultiScaleSetupY.SetupDuals();
    if(msg!=0) { printf("error: %d\n",msg); out.msg = msg; return msg; }
    int nLayers = MultiScaleSetupX.nLayers;


    Familys<OT_PROBLEM_SIZE(image_size),GETLOG2(image_size)> * familyX =
            createFamily<OT_PROBLEM_SIZE(image_size),GETLOG2(image_size)>(&MultiScaleSetupX);
    Familys<OT_PROBLEM_SIZE(image_size),GETLOG2(image_size)> * familyY =
            createFamily<OT_PROBLEM_SIZE(image_size),GETLOG2(image_size)>(&MultiScaleSetupY);
    familyX->check_for_layers(&MultiScaleSetupX);


//  epsScaling
    double epsStart=1E8;
    double epsTarget=1E-1;
    int epsSteps=20;
    TEpsScalingHandler epsScalingHandler(epsStart,epsTarget,epsSteps); // basic eps scaling
    epsScalingHandler.getEpsScalesFromBox(muXdim[0],2,MultiScaleSetupX.nLayers); // eps scales for each layer
    epsScalingHandler.getEpsScalingSplit(layerCoarsest,1); // create sub eps lists for each layer
    printf("total layers = %d\n",MultiScaleSetupX.nLayers);
    CuEspsScalingHandler_ h_eps;
    h_eps.init_eps(epsScalingHandler);


    CuSinkhornSolverParameters cfg_={
        1E-4, // maxError
        100000, // maxIterations
        1000, // innerIterations
        20, // maxAbsorptionLoops
        1E3, // absorption_scalingBound
        1E3, // absorption_scalingLowerBound
        1E-1, // truncation_thresh
        1 // refineKernel when refining layer (as opposed to attempting to estimate directly)
        };


    // upstream: solver.cu:1324-1432
     int *numsX = (int *)malloc(sizeof(int)*nLayers);
     int *numsY = (int *)malloc(sizeof(int)*nLayers);
     for (int i = 0 ; i < nLayers ; ++i){
         numsX[i] = MultiScaleSetupX.HP->layers[i]->nCells;
         numsY[i] = MultiScaleSetupY.HP->layers[i]->nCells;
     }

     int *numsradiiX = (int *)malloc(sizeof(int)*nLayers);
     int *numsradiiY = (int *)malloc(sizeof(int)*nLayers);
     for (int i = 0 ; i < nLayers -2 ; ++i){
         numsradiiX[i] = MultiScaleSetupX.HB->layers[i].nodes.size();
         numsradiiY[i] = MultiScaleSetupY.HB->layers[i].nodes.size();
     }

     int *numposX = (int *)malloc(sizeof(int)*nLayers);
     int *numposY = (int *)malloc(sizeof(int)*nLayers);
     for (int i = 0 ; i < nLayers ; ++i){
         numposX[i] = MultiScaleSetupX.HB->layers[i].nodes.size()*2;
         numposY[i] = MultiScaleSetupY.HB->layers[i].nodes.size()*2;
     }

     XData *xposH = create_dev_struct<XData>(numposX,nLayers,MultiScaleSetupX.posH);
     XData *cu_xposH =  malloc_struct<XData>(xposH);
     XData *yposH = create_dev_struct<XData>(numposY,nLayers,MultiScaleSetupY.posH);
     XData *cu_yposH = malloc_struct<XData>(yposH);

     XData *xradii = create_dev_struct<XData>(numsradiiX,nLayers-2,MultiScaleSetupX.radii);
     XData *cu_xradii =  malloc_struct<XData>(xradii);

     XData *yradii = create_dev_struct<XData>(numsradiiY,nLayers-2,MultiScaleSetupY.radii);
     XData *cu_yradii = malloc_struct<XData>(yradii);


     XData *muXH = create_dev_struct<XData>(numsX,nLayers,MultiScaleSetupX.muH);
     XData *cu_muXH =  malloc_struct<XData>(muXH);
     XData *muYH = create_dev_struct<XData>(numsY,nLayers,MultiScaleSetupY.muH);
     XData *cu_muYH = malloc_struct<XData>(muYH);

     XData *alphaH = create_dev_struct<XData>(numsX,nLayers,MultiScaleSetupX.alphaH);
     XData *cu_alphaH =  malloc_struct<XData>(alphaH);
     XData *betaH = create_dev_struct<XData>(numsY,nLayers,MultiScaleSetupY.alphaH);
     XData *cu_betaH =  malloc_struct<XData>(betaH);




     Ind_Ptr_ *indptrl = (Ind_Ptr_ *)malloc(sizeof(Ind_Ptr_) * kMaxEl);
     for (int i = 0 ;  i < kMaxEl; ++i){
         indptrl[i] = i;
     }
     Ind_Ptr_ *indptrl_cu_mid = malloc_struct(indptrl,kMaxEl);
     Ind_Ptr_ *indptrl_cu;
     CHECK(cudaMalloc((void **)&indptrl_cu, sizeof(Ind_Ptr_) * kMaxEl ));

    Familys<OT_PROBLEM_SIZE(image_size),GETLOG2(image_size)> *cu_familyX,*cu_familyY;
    size_t cufamily_size = sizeof(Familys<OT_PROBLEM_SIZE(image_size),GETLOG2(image_size)>);
    CHECK(cudaMalloc((void **)&cu_familyX, cufamily_size));
    CHECK(cudaMalloc((void **)&cu_familyY, cufamily_size));
    CHECK(cudaMemcpy(cu_familyX, familyX, cufamily_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(cu_familyY, familyY, cufamily_size, cudaMemcpyHostToDevice));

    COO_Buff * buff = NULL;   // passed through, never dereferenced (upstream leaves it uninitialized)
    double *muX_cuda ;
    double *muY_cuda ;

    double *u;
    double *v;
    double *res;
    double * origonals_one_cuda;
    CHECK(cudaMalloc((void **)&origonals_one_cuda,sizeof(double)*OT_PROBLEM_SIZE(image_size)));

    CHECK(cudaMalloc((void **)&muX_cuda,sizeof(double)*OT_PROBLEM_SIZE(image_size)));
    CHECK(cudaMalloc((void **)&muY_cuda,sizeof(double)*OT_PROBLEM_SIZE(image_size)));
    CHECK(cudaMalloc((void **)&u,sizeof(double)*OT_PROBLEM_SIZE(image_size)));
    CHECK(cudaMalloc((void **)&v,sizeof(double)*OT_PROBLEM_SIZE(image_size)));
    CHECK(cudaMalloc((void **)&res,sizeof(double)*OT_PROBLEM_SIZE(image_size)));


    int layer = layerCoarsest;
    int kernelValid = 0;
    int xres = 0;
    int yres = 0;
    int layerBottom = layerFinest;
    double * origionals = (double *)malloc(sizeof(double) * OT_PROBLEM_SIZE(image_size));
    for (int i = 0 ; i < OT_PROBLEM_SIZE(image_size); ++i)
        origionals[i] = 1.0;
    CHECK(cudaMemcpy(origonals_one_cuda, origionals, sizeof(double)*OT_PROBLEM_SIZE(image_size),
                     cudaMemcpyHostToDevice));
    int *msg_cu;
    cudaMalloc((void **)&msg_cu,sizeof(int));
    msg = changeLayer(layerCoarsest,&layer,&kernelValid,
                MultiScaleSetupX.HP,MultiScaleSetupY.HP,
                &xres,&yres,u,v,origonals_one_cuda,
                           muX_cuda,muY_cuda,
                           cu_muXH,cu_muYH,muXH,muYH);
    COO_Sparese_Matrix<kMaxEl> *kernel,*kernelT;
    CHECK(cudaMalloc((void **)&kernel,sizeof(COO_Sparese_Matrix<kMaxEl>)));
    CHECK(cudaMalloc((void **)&kernelT,sizeof(COO_Sparese_Matrix<kMaxEl>)));
    double eps = 1e10;
    if(msg!=0) { out.msg = msg; return msg; }

    cublasHandle_t handle;
    cublasCreate(&handle);

    // first-party
    std::vector<long long> iters_by_layer(nLayers, 0);

    //generate the whole region (upstream: solver.cu:1439-1517)
    msg=generateKernel_<kMaxEl>(kernel,kernelT,&cfg_,eps,
                    muX_cuda,muY_cuda,layer,
                    cu_alphaH,cu_betaH,cu_xposH,cu_yposH,cu_xradii,cu_yradii,xres,yres,
                    indptrl_cu_mid,indptrl_cu,buff);
    kernelValid = 1;




    while(true) {

        if( layer >= 8){
            // for large problems
            cfg_.maxError = 1E-4;
        }
        for(int nEps=0; nEps<h_eps.nEpsLists[layer]; nEps++) {
            eps = h_eps.epsLists[layer][nEps];
            printf("\teps=%e\n",eps);
            kernelValid = 0;
            if(cfg_.refineKernel && (nEps==0) && (layer>layerCoarsest)) {
                refinekernel_<image_size,image_size,GETLOG2(image_size),kMaxEl>(kernel,kernelT,muX_cuda,
                                  muY_cuda,
                                  layerBottom,
                                  eps,xres,yres,
                                  layer,
                                  cu_familyX,cu_familyY,
                                  cu_alphaH,cu_betaH,
                                  cu_xposH, cu_yposH,
                                  cu_xradii, cu_yradii,
                                 indptrl_cu_mid,indptrl_cu,
                                 &kernelValid,buff);


            }
            int msg=shim_solveSingle_<image_size,image_size,GETLOG2(image_size),kMaxEl>(msg_cu,&kernelValid,
                                kernel,kernelT,
                                &cfg_,eps,
                                muX_cuda,muY_cuda,
                                cu_familyX,cu_familyY,
                                layerBottom,
                                cu_alphaH,cu_betaH,
                                cu_xposH,cu_yposH,
                                cu_xradii,cu_yradii,
                                xres,yres,u,v,res,layer,
                                indptrl_cu_mid,
                                indptrl_cu,buff,
                                 handle,
                                 &iters_by_layer[layer]
                                );

            if(msg!=0) { out.msg = msg; return msg; }
        }



        if(msg!=0) { out.msg = msg; return msg; }
        if(layer < layerFinest) {


            int newLayer = layer + 1;
            if(newLayer>0) {
                signal_refine_all<OT_PROBLEM_SIZE(image_size),GETLOG2(image_size)>(cu_familyX,cu_familyY,cu_alphaH,cu_betaH,layer);
            }
            kernelValid=0;
            if(msg!=0) { out.msg = msg; return msg; }

            msg = changeLayer(newLayer,&layer,&kernelValid,
                        MultiScaleSetupX.HP,MultiScaleSetupY.HP,
                        &xres,&yres,u,v,origonals_one_cuda,
                                   muX_cuda,muY_cuda,
                                   cu_muXH,cu_muYH,muXH,muYH
                             );

            if(msg!=0) { out.msg = msg; return msg; }

        } else {
            break;
        }

    }


    // delivery rebuild (upstream: solver.cu:1522-1536)
    int N = 0;
    msg = refinekernel_2<kMaxEl>(kernel,kernelT,
                      muX_cuda,
                      muY_cuda,
                      layerBottom,
                      eps,xres,yres,
                      layer,
                      cu_alphaH,cu_betaH,cu_xposH,cu_yposH,cu_xradii,cu_yradii,
                     indptrl_cu_mid,indptrl_cu,
                      &kernelValid,buff,&cfg_);

    cudaMemcpy(&N,&kernel->nonZeros,sizeof(int),cudaMemcpyDeviceToHost);

    printf("return code: %d\n",msg);
    if(msg!=0) { out.msg = msg; return msg; }

    SHIM_CHECK(cudaDeviceSynchronize());           // first-party timing fence
    double t_solve1 = seconds();

    // First-party: copy the plan and the duals out. The copies must precede
    // calculate_result, whose scoreTransportCost multiplies kernel->val by the
    // cost in place (solver.cu:747). kernel is a device pointer whose members are
    // arrays, so kernel->row_ptrl etc. only form device addresses, as upstream
    // does (solver.cu:242, 1097).
    double t_ext0 = seconds();
    std::vector<int> h_row, h_col;
    std::vector<double> h_val, h_u, h_v, h_alpha, h_beta;
    if (outdir != nullptr) {
        h_row.resize(N); h_col.resize(N); h_val.resize(N);
        h_u.resize(xres); h_v.resize(yres);
        h_alpha.resize(xres); h_beta.resize(yres);
        SHIM_CHECK(cudaMemcpy(h_row.data(), kernel->row_ptrl, sizeof(int)*N,    cudaMemcpyDeviceToHost));
        SHIM_CHECK(cudaMemcpy(h_col.data(), kernel->col_ptrl, sizeof(int)*N,    cudaMemcpyDeviceToHost));
        SHIM_CHECK(cudaMemcpy(h_val.data(), kernel->val,      sizeof(double)*N, cudaMemcpyDeviceToHost));
        SHIM_CHECK(cudaMemcpy(h_u.data(), u, sizeof(double)*xres, cudaMemcpyDeviceToHost));
        SHIM_CHECK(cudaMemcpy(h_v.data(), v, sizeof(double)*yres, cudaMemcpyDeviceToHost));
        // Offsets from the host mirrors alphaH, betaH (never freed), whose ind[]
        // equals the device structs'.
        SHIM_CHECK(cudaMemcpy(h_alpha.data(), &cu_alphaH->data[alphaH->ind[layerFinest]],
                              sizeof(double)*xres, cudaMemcpyDeviceToHost));
        SHIM_CHECK(cudaMemcpy(h_beta.data(),  &cu_betaH->data[betaH->ind[layerFinest]],
                              sizeof(double)*yres, cudaMemcpyDeviceToHost));
        // π_k = val · u · v as upstream scores it; u = v = 1 here, so π_k = val[k].
        for (int k = 0; k < N; ++k)
            h_val[k] = h_val[k] * h_u[h_row[k]] * h_v[h_col[k]];
    }

    // The distance upstream returns (solver.cu:1550); mutates kernel->val.
    double result = calculate_result<kMaxEl>(N,cu_xposH,cu_yposH,res,u,v,kernel,layer);
    SHIM_CHECK(cudaDeviceSynchronize());
    double t_ext1 = seconds();

    if (outdir != nullptr) {
        if (!write_blob(*outdir + "/plan_row.bin", h_row.data(), sizeof(int)*N) ||
            !write_blob(*outdir + "/plan_col.bin", h_col.data(), sizeof(int)*N) ||
            !write_blob(*outdir + "/plan_val.bin", h_val.data(), sizeof(double)*N) ||
            !write_blob(*outdir + "/alpha.bin", h_alpha.data(), sizeof(double)*xres) ||
            !write_blob(*outdir + "/beta.bin",  h_beta.data(),  sizeof(double)*yres)) {
            out.msg = -10;   // blob-write failure, not an upstream msg
            return -10;
        }
    }

    out.msg = 0;
    out.distance_native = result;
    out.eps_final = eps;
    out.solve_seconds = t_solve1 - t_solve0;
    out.extract_seconds = t_ext1 - t_ext0;
    out.nnz = N;
    out.xres_final = xres;
    out.n_layers = nLayers;
    out.iters_per_layer = iters_by_layer;

    // upstream: solver.cu:1599-1624
    cublasDestroy(handle);
    cudaFree(msg_cu);
    cudaFree(muX_cuda);
    cudaFree(muY_cuda);
    cudaFree(u);
    cudaFree(v);
    cudaFree(kernel);
    cudaFree(kernelT);
    cudaFree(indptrl_cu);
    cudaFree(indptrl_cu_mid);
    cudaFree(cu_xposH);
    cudaFree(cu_yposH);
    cudaFree(cu_betaH);
    cudaFree(cu_alphaH);
    cudaFree(cu_xradii);
    cudaFree(cu_yradii);
    cudaFree(res);
    cudaFree(cu_muXH);
    cudaFree(cu_muYH);
    cudaFree(origonals_one_cuda);



    cudaFree(cu_familyX);
    cudaFree(cu_familyY);

    // Upstream leaks its host buffers and calculate_result's cuBLAS handle on
    // every solve; a shim process solves at most twice.

    return 0;
}

// Warm-up marginals, never instance data: two Gaussian blobs plus a small
// seeded perturbation, strictly positive (the multiscale setup rejects
// nonpositive mass) and of unit mass.
static void make_warmup_pair(int ng, std::vector<double> &a, std::vector<double> &b) {
    a.assign((size_t)ng * ng, 0.0);
    b.assign((size_t)ng * ng, 0.0);
    uint64_t lcg = 20260427u;
    auto noise = [&lcg]() {
        lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
        return (double)(lcg >> 11) / (double)(1ULL << 53);
    };
    const double sa = 0.15 * ng, sb = 0.20 * ng;
    double suma = 0.0, sumb = 0.0;
    for (int i = 0; i < ng; ++i) {
        for (int j = 0; j < ng; ++j) {
            double da = (i - 0.35 * ng) * (i - 0.35 * ng) + (j - 0.30 * ng) * (j - 0.30 * ng);
            double db = (i - 0.60 * ng) * (i - 0.60 * ng) + (j - 0.70 * ng) * (j - 0.70 * ng);
            double va = exp(-da / (2.0 * sa * sa)) + 1e-6 * noise() + 1e-9;
            double vb = exp(-db / (2.0 * sb * sb)) + 1e-6 * noise() + 1e-9;
            a[(size_t)i * ng + j] = va;  suma += va;
            b[(size_t)i * ng + j] = vb;  sumb += vb;
        }
    }
    for (auto &x : a) x /= suma;
    for (auto &x : b) x /= sumb;
}

static bool read_blob(const std::string &path, std::vector<double> &out, size_t n) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "m3s_shim: cannot open %s\n", path.c_str()); return false; }
    out.resize(n);
    size_t r = fread(out.data(), sizeof(double), n, f);
    fclose(f);
    if (r != n) {
        fprintf(stderr, "m3s_shim: %s holds %zu doubles, expected %zu\n", path.c_str(), r, n);
        return false;
    }
    return true;
}

static int dispatch(int ng, std::vector<double> &a, std::vector<double> &b,
                    const std::string *outdir, ShimResult &res) {
    // Upstream's image_ot_solver switch (image_ot_solver.cu:22-40) plus a 1024
    // case, which main() refuses before dispatching.
    switch (ng) {
    case 32:   return shim_solve<32>(a, b, outdir, res);
    case 64:   return shim_solve<64>(a, b, outdir, res);
    case 128:  return shim_solve<128>(a, b, outdir, res);
    case 256:  return shim_solve<256>(a, b, outdir, res);
    case 512:  return shim_solve<512>(a, b, outdir, res);
    case 1024: return shim_solve<1024>(a, b, outdir, res);
    default:
        fprintf(stderr, "m3s_shim: unsupported image_size %d\n", ng);
        return -1;
    }
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr,
                "usage: m3s_shim <n_g> <r.bin> <c.bin> <outdir> [--warmup]\n"
                "  r.bin/c.bin: raw fp64 vectors of length n_g^2 (row-major"
                " probability marginals)\n");
        return 1;
    }
    const int ng = atoi(argv[1]);
    const std::string r_path = argv[2];
    const std::string c_path = argv[3];
    const std::string outdir = argv[4];
    bool warmup = false;
    for (int i = 5; i < argc; ++i)
        if (strcmp(argv[i], "--warmup") == 0) warmup = true;

    if (ng != 32 && ng != 64 && ng != 128 && ng != 256 && ng != 512 && ng != 1024) {
        fprintf(stderr, "m3s_shim: n_g must be one of 32/64/128/256/512/1024; got %d\n", ng);
        return 1;
    }
    if (ng == 1024) {
        fprintf(stderr,
                "m3s_shim: n_g=1024 is REFUSED: the as-published fixed-size "
                "structs (XData data[512^2*3], crs_row_ptrl[512^2+1]; "
                "MAXSIZE=512 in Common/basic_settings.h) are overflowed by the "
                "1024^2 atomic hierarchy layer, which is memory-unsafe "
                "without upstream edits.\n");
        return 1;
    }

    const size_t N = (size_t)ng * ng;
    std::vector<double> r_vec, c_vec;
    if (!read_blob(r_path, r_vec, N) || !read_blob(c_path, c_vec, N)) return 1;

    if (warmup) {
        fprintf(stderr, "m3s_shim: warmup solve (generic dummy pair, n_g=%d)\n", ng);
        std::vector<double> wa, wb;
        make_warmup_pair(ng, wa, wb);
        ShimResult wres;
        int wmsg = dispatch(ng, wa, wb, nullptr, wres);
        if (wmsg != 0)
            fprintf(stderr, "m3s_shim: warmup returned msg=%d (non-fatal)\n", wmsg);
        cudaDeviceSynchronize();
    }

    ShimResult res;
    int msg = dispatch(ng, r_vec, c_vec, &outdir, res);

    // Written after every solve, failed or not: the runner reads msg from it
    // on exit 3.
    {
        std::string meta_path = outdir + "/meta.json";
        FILE *f = fopen(meta_path.c_str(), "w");
        if (!f) { fprintf(stderr, "m3s_shim: cannot open %s\n", meta_path.c_str()); return 1; }
        fprintf(f, "{\n  \"msg\": %d,\n", res.msg);
        fprintf(f, "  \"distance_native\": %.17g,\n", res.distance_native);
        fprintf(f, "  \"eps_final\": %.17g,\n", res.eps_final);
        fprintf(f, "  \"shim_solve_seconds\": %.6f,\n", res.solve_seconds);
        fprintf(f, "  \"shim_extract_seconds\": %.6f,\n", res.extract_seconds);
        fprintf(f, "  \"nnz\": %lld,\n", res.nnz);
        fprintf(f, "  \"xres_final\": %d,\n", res.xres_final);
        fprintf(f, "  \"n_layers\": %d,\n", res.n_layers);
        fprintf(f, "  \"warmup\": %s,\n", warmup ? "true" : "false");
        fprintf(f, "  \"iters_per_layer\": [");
        for (size_t i = 0; i < res.iters_per_layer.size(); ++i)
            fprintf(f, "%s%lld", i ? ", " : "", res.iters_per_layer[i]);
        fprintf(f, "]\n}\n");
        fclose(f);
    }

    if (msg == -10) {
        fprintf(stderr, "m3s_shim: output blob write failed (not a solver msg)\n");
        return 1;
    }
    if (msg != 0) {
        fprintf(stderr, "m3s_shim: solver msg=%d (30101=max iters, 30201=too "
                        "many absorptions, -2=nan error)\n", msg);
        return 3;
    }
    printf("m3s_shim: ok nnz=%lld distance_native=%.12g solve=%.3fs\n",
           res.nnz, res.distance_native, res.solve_seconds);
    return 0;
}
