
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<time.h>
#include"dt_mn_fncs.h"
#include"trn_fncs.h"

int main(void)
{
    int i, j, k, buffer_size = 65536, train_rows, test_rows;
    int rows, columns, classes, features, *meta_data;
    int num_hidden_layers, *layer_sizes;
    double **input_data;
    double rand_min = -1.0, rand_max = 1.0;
    double learning_rate, l2_lambda;
    int epochs;
    double test_acc;
    char path[buffer_size];
    const char *delimiter = ",";
    double split;
    int random_seed;
    MLP *mlp;

    printf("Insert the file name: ");
    scanf("%s", path);
    printf("Insert the number of hidden layers: ");
    scanf("%d", &num_hidden_layers);
    
    layer_sizes = malloc_1D_int(num_hidden_layers);
    for(i=0; i<num_hidden_layers; i++)
    {
        printf("Insert the number of neurons in layer %d: ", i+1);
        scanf("%d", &layer_sizes[i]);
    }
    
    printf("Insert the number of epochs: ");
    scanf("%d", &epochs);
    printf("Insert the learning rate: ");
    scanf("%lf", &learning_rate);
    printf("Insert L2 regularization lambda (0 for none, typical: 0.0001-0.01): ");
    scanf("%lf", &l2_lambda);
    printf("Insert the train set size (%%): ");
    scanf("%lf", &split);
    printf("Insert random seed (for reproducible results): ");
    scanf("%d", &random_seed);
    
    srand(random_seed);

    printf("Loading dataset metadata...\n");
    fflush(stdout);
    meta_data = get_meta_data_from_csv(path, delimiter, buffer_size);
	
    if(meta_data != NULL)
    {
        rows = meta_data[0];
        columns = meta_data[1];
        classes = meta_data[2];
        features = meta_data[3];
        
        print_meta_data(meta_data);
        
        size_t dataset_memory = (size_t)rows * columns * sizeof(double) + rows * sizeof(double*);
        size_t mlp_memory = (size_t)(features * layer_sizes[0] + layer_sizes[num_hidden_layers-1] * classes) * sizeof(double) * 2;
        size_t total_estimated = dataset_memory + mlp_memory;
        
        printf("Estimated memory usage: %.2f MB\n", total_estimated / (1024.0 * 1024.0));
        
        if(total_estimated > 2UL * 1024 * 1024 * 1024)
        {
            printf("WARNING: Dataset is very large (>2GB). This may cause memory issues.\n");
        }
        
        printf("Reading dataset from CSV...\n");
        fflush(stdout);
        input_data = read_from_csv(path, buffer_size, rows, columns);
        printf("Normalizing dataset...\n");
        fflush(stdout);
        normalize_dataset(input_data, rows, columns);

        shuffle(input_data, rows);
        train_rows = (int)(split*rows);
        test_rows = rows-train_rows;

        printf("Train samples: %d, Test samples: %d\n", train_rows, test_rows);

        printf("Creating MLP model...\n");
        fflush(stdout);
        mlp = create_mlp(features, classes, num_hidden_layers, layer_sizes, learning_rate, l2_lambda, rand_min, rand_max);
        if(!mlp) 
        {
            printf("Error: Failed to create MLP model - not enough memory\n");
            free_matrix(input_data, rows);
            free(meta_data);
            free(layer_sizes);
            return 1;
        }

        train_mlp(mlp, input_data, train_rows, columns, epochs);

        printf("Samples used in training:\n");
        for(i=0; i<train_rows; i++)
        {
            printf("Sample %d:", i);
            for(j=0; j<features; j++)
            {
            	 printf(" f%d: %.3lf -", j, input_data[i][j]);
            }
            printf(" c:%d\n", (int)input_data[i][j]);
        }

        test_acc = test_model(mlp, input_data, train_rows, rows, columns, classes);
        printf("\nTest accuracy: %.2f%% (%d/%d correct)\n", test_acc, (int)(test_acc*test_rows/100.0), test_rows);

        free_matrix(input_data, rows);
        free_mlp(mlp);
        free(meta_data);
        free(layer_sizes);
    }

    return 0;
}
