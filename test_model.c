#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<time.h>
#include"dt_mn_fncs.h"
#include"trn_fncs.h"

int main(void)
{
    const int buffer_size = 65536;
    char dataset_path[buffer_size];
    char model_path[buffer_size];
    const char *delimiter = ",";
    int random_seed = time(NULL);

    printf("Enter dataset CSV path: ");
    scanf("%s", dataset_path);

    printf("Enter model binary path: ");
    scanf("%s", model_path);

    int *meta = get_meta_data_from_csv(dataset_path, delimiter, buffer_size);
    if(!meta) {
        printf("Failed to read dataset metadata\n");
        return 1;
    }

    int rows = meta[0];
    int columns = meta[1];
    int classes = meta[2];
    int features = meta[3];

    double **data = read_from_csv(dataset_path, buffer_size, rows, columns);
    if(!data) {
        printf("Failed to read dataset\n");
        free(meta);
        return 1;
    }

    normalize_dataset(data, rows, columns);

    MLP *mlp = load_mlp_binary(model_path);
    if(!mlp) {
        printf("Failed to load model\n");
        free_matrix(data, rows);
        free(meta);
        return 1;
    }

    // Test on whole dataset
    double acc = test_model(mlp, data, 0, rows, columns, classes);
    printf("\nModel test accuracy: %.2f%%\n", acc);

    free_matrix(data, rows);
    free_mlp(mlp);
    free(meta);

    return 0;
}
