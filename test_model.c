#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include<math.h>
#include"dt_mn_fncs.h"
#include"trn_fncs.h"

int main (void){
    const int buffer_size = 65536;
    int random_seed = time(NULL);
    char model_path[buffer_size];
    MLP *mlp = NULL;

    printf("Enter model binary path: ");
    scanf("%s", model_path);

    mlp = load_mlp_binary(model_path);
    if(!mlp) {
        printf("Failed to load model\n");
        return 1;
    }

    // Hardcoded sample features for inference (no class value)
    // Modify these values to match your use-case.
double sample_features[] = {12,3.414949417114258,1.757750153541565,9.833800315856934,9.270182609558105,8.23648738861084,7.712896347045898,7.508970260620117,7.405328273773193,7.64684534072876,7.644525527954102,7.839168071746826,8.019719123840332,8.239964485168457,8.350132942199707,8.277363777160645,8.395174980163574,8.379159927368164,8.641219139099121,1.6433411836624146,1.8976876735687256,1.9824484586715698,1.6425604820251465,0.9427992701530457,0.4605252742767334,0.548456609249115,0.32453909516334534,0.8414579033851624,0.6318063139915466,0.3247518837451935,0.2712450325489044,0.312822550535202,0.3674841821193695,0.40661704540252686,0.36899662017822266,0.8275021910667419,1.0105793476104736,1.0586774349212646,1.2401858568191528,1.0267329216003418,0.4173612892627716,0.5769218802452087,0.45834943652153015,0.8957334756851196,0.8479160070419312,0.40341994166374207,0.304031640291214,0.3861698806285858,0.48967209458351135,0.5404180288314819,0.4564959704875946};
    int feature_count = sizeof(sample_features) / sizeof(sample_features[0]);
    // Prepare input vector matching the model's expected feature size
    int model_features = mlp->features;
    int classes = mlp->classes;

    if(feature_count != model_features) {
        printf("WARNING: model expects %d features but sample has %d.\n", model_features, feature_count);
        // We'll copy what we can and pad with zeros if necessary.
    }

    double *input_vec = malloc_1D(model_features);
    if(!input_vec) {
        printf("Failed to allocate input vector\n");
        free_mlp(mlp);
        return 1;
    }

    // Copy sample values (truncate or pad with 0.0)
    for(int i=0; i<model_features; i++) {
        if(i < feature_count) input_vec[i] = sample_features[i];
        else input_vec[i] = 0.0;
    }

    // NOTE: If the model was trained with normalized inputs, provide normalized
    // features here. This code assumes the hardcoded sample is already in the
    // correct scale expected by the model.

    // Run inference
    mlp_feedforward(mlp, input_vec);
    int predicted = arg_max(mlp->outputs[mlp->num_hidden_layers], classes);
    printf("Predicted class for single sample: %d\n", predicted);

    // Print output probabilities
    printf("Output probabilities:");
    for(int c=0; c<classes; c++) {
        printf(" %.4f", mlp->outputs[mlp->num_hidden_layers][c]);
    }
    printf("\n");

    free(input_vec);
    free_mlp(mlp);

    return 0;
}
