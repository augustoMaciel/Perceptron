/*
 * nnfncs.h
 *
 *  Created on: 11 de jun. de 2025
 *      Author: Augusto Lipinski Fernandes Maciel
 */

#ifndef TRN_FNCS_H_
#define TRN_FNCS_H_
#include "dt_mn_fncs.h"
#include "nn_fncs.h"
#include "mtrcs.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#define MAX_GRADIENT_CLIP 10.0
#define MIN_LEARNING_RATE 1e-8

typedef struct 
{
    int features;
    int classes;
    int num_hidden_layers;
    int *layer_sizes;
    double ***weights;
    double **outputs;
    double learning_rate;
    double l2_lambda;
} 
MLP;

void free_mlp(MLP *mlp);
void free_matrix_contiguous(double **matrix, int rows);
int save_mlp_binary(MLP *mlp, const char *path);
MLP* load_mlp_binary(const char *path);

void init_weights (double **matrix, int rows, int columns, double max, double min)
{
    int i, j;
    double range = max - min;
    
    for(i=0; i<rows; i++)
    {
        for(j=0; j<columns; j++)
        {
            matrix[i][j] = min+range*((double)rand()/RAND_MAX);
        }
    }
}

void init_weights_xavier(double **matrix, int rows, int columns)
{
    int i, j;
    double variance = 2.0 / (rows + columns - 1);
    double limit = sqrt(3.0 * variance);
    
    for(i=0; i<rows; i++)
    {
        for(j=0; j<columns; j++)
        {
            matrix[i][j] = -limit + 2.0*limit*((double)rand()/RAND_MAX);
        }
    }
}

void init_weights_he(double **matrix, int rows, int columns)
{
    int i, j;
    int fan_in = columns - 1;
    if(fan_in <= 0) fan_in = 1;

    double variance = 2.0 / (double)fan_in;
    double limit = sqrt(3.0 * variance);

    for(i=0; i<rows; i++)
    {
        for(j=0; j<columns; j++)
        {
            matrix[i][j] = -limit + 2.0*limit*((double)rand()/RAND_MAX);
        }
    }
}

MLP* create_mlp(int features, int classes, int num_hidden_layers, int *layer_sizes, double learning_rate, double l2_lambda, double rand_min, double rand_max)
{
    int i, input_size, output_size;
    
    if(features <= 0 || classes <= 0 || num_hidden_layers <= 0 || !layer_sizes) 
    {
        printf("ERROR: Invalid MLP parameters\n");
        return NULL;
    }
    
    printf("  Allocating MLP structure...\n"); fflush(stdout);
    MLP *mlp = (MLP*)malloc(sizeof(MLP));
    if(!mlp) return NULL;
    
    mlp->features = features;
    mlp->classes = classes;
    mlp->num_hidden_layers = num_hidden_layers;
    mlp->learning_rate = learning_rate;
    mlp->l2_lambda = l2_lambda;
    
    printf("  Allocating layer_sizes array...\n"); fflush(stdout);
    mlp->layer_sizes = malloc_1D_int(num_hidden_layers);
    if(!mlp->layer_sizes) 
    {
        printf("  ERROR: Failed to allocate layer_sizes\n");
        free(mlp);
        return NULL;
    }
    for(i=0; i<num_hidden_layers; i++)
    {
        mlp->layer_sizes[i] = layer_sizes[i];
    }
    
    printf("  Allocating weights array...\n"); fflush(stdout);
    mlp->weights = malloc_3D_partial(num_hidden_layers + 1);
    if(!mlp->weights) 
    {
        printf("  ERROR: Failed to allocate weights array\n");
        free(mlp->layer_sizes);
        free(mlp);
        return NULL;
    }
    
    printf("  Allocating outputs array...\n"); fflush(stdout);
    mlp->outputs = malloc_2D_partial(num_hidden_layers + 1);
    if(!mlp->outputs) 
    {
        printf("  ERROR: Failed to allocate outputs array\n");
        free(mlp->weights);
        free(mlp->layer_sizes);
        free(mlp);
        return NULL;
    }
    
    input_size = features;
    output_size = layer_sizes[0];
    printf("  Allocating first layer: %d x %d weights...\n", output_size, input_size + 1); fflush(stdout);
    mlp->weights[0] = malloc_2D_complete(output_size, input_size + 1);
    mlp->outputs[0] = malloc_1D(output_size);
    if(!mlp->weights[0] || !mlp->outputs[0]) 
    {
        printf("  ERROR: Failed to allocate first layer\n");
        free_mlp(mlp);
        return NULL;
    }
    printf("  Initializing first layer weights...\n"); fflush(stdout);
    init_weights_he(mlp->weights[0], output_size, input_size + 1);
    
    for(i=1; i<num_hidden_layers; i++)
    {
        input_size = layer_sizes[i-1];
        output_size = layer_sizes[i];
        printf("  Allocating hidden layer %d: %d x %d weights...\n", i, output_size, input_size + 1); fflush(stdout);
        mlp->weights[i] = malloc_2D_complete(output_size, input_size + 1);
        mlp->outputs[i] = malloc_1D(output_size);
        if(!mlp->weights[i] || !mlp->outputs[i]) 
        {
            printf("  ERROR: Failed to allocate hidden layer %d\n", i);
            free_mlp(mlp);
            return NULL;
        }
        printf("  Initializing hidden layer %d weights...\n", i); fflush(stdout);
        init_weights_he(mlp->weights[i], output_size, input_size + 1);
    }
    
    input_size = layer_sizes[num_hidden_layers - 1];
    output_size = classes;
    printf("  Allocating output layer: %d x %d weights...\n", output_size, input_size + 1); fflush(stdout);
    mlp->weights[num_hidden_layers] = malloc_2D_complete(output_size, input_size + 1);
    mlp->outputs[num_hidden_layers] = malloc_1D(output_size);
    if(!mlp->weights[num_hidden_layers] || !mlp->outputs[num_hidden_layers]) 
    {
        printf("  ERROR: Failed to allocate output layer\n");
        free_mlp(mlp);
        return NULL;
    }
    printf("  Initializing output layer weights...\n"); fflush(stdout);
    init_weights_he(mlp->weights[num_hidden_layers], output_size, input_size + 1);
    
    printf("  MLP creation completed successfully!\n"); fflush(stdout);
    return mlp;
}

void free_mlp(MLP *mlp)
{
    int i;
    
    if(!mlp) return;
    
    if(mlp->weights) 
    {
        for(i=0; i<=mlp->num_hidden_layers; i++)
        {
            if(mlp->weights[i]) 
            {
                if(i==0)
                    free_matrix_contiguous(mlp->weights[i], mlp->layer_sizes[0]);
                else if(i<mlp->num_hidden_layers)
                    free_matrix_contiguous(mlp->weights[i], mlp->layer_sizes[i]);
                else
                    free_matrix_contiguous(mlp->weights[i], mlp->classes);
            }
        }
        free(mlp->weights);
    }
    
    if(mlp->outputs) {
        for(i=0; i<=mlp->num_hidden_layers; i++)
        {
            if(mlp->outputs[i]) {
                free(mlp->outputs[i]);
            }
        }
        free(mlp->outputs);
    }
    
    if(mlp->layer_sizes) {
        free(mlp->layer_sizes);
    }
    free(mlp);
}

double* one_hot(int int_class, int classes) 
{
	int i;
    double* vector = malloc_1D(classes);
    if(!vector) return NULL;
	
    for(i=0; i<classes; i++)
	{
        vector[i] = 0.0;
	}
    vector[int_class] = 1.0;
	
    return vector;
}

void forward_pass (double *input, double **weights, int rows, int columns, double *result)
{
	int i, j;
	double sum = 0, bias;
	
	for(i=0; i<rows; i++)
	{
		bias = weights[i][columns-1];
		for(j=0; j<columns-1; j++)
		{
			sum += input[j]*weights[i][j];
		}
		result[i] = sum+bias;
		sum = 0;
	}
}

double calculate_l2_loss(MLP *mlp)
{
    int i, j, k, layer;
    double l2_loss = 0.0;
    
    for(layer=0; layer<=mlp->num_hidden_layers; layer++)
    {
        int rows, cols;
        
        if(layer == 0)
        {
            rows = mlp->layer_sizes[0];
            cols = mlp->features;
        }
        else if(layer < mlp->num_hidden_layers)
        {
            rows = mlp->layer_sizes[layer];
            cols = mlp->layer_sizes[layer-1];
        }
        else
        {
            rows = mlp->classes;
            cols = mlp->layer_sizes[mlp->num_hidden_layers-1];
        }
        
        for(i=0; i<rows; i++)
        {
            for(j=0; j<cols; j++)
            {
                l2_loss += mlp->weights[layer][i][j] * mlp->weights[layer][i][j];
            }
        }
    }
    
    return (mlp->l2_lambda / 2.0) * l2_loss;
}

void weight_update(MLP *mlp, double *input, double **deltas)
{
    int j, k, layer;
    
    int last_hidden_size = mlp->layer_sizes[mlp->num_hidden_layers - 1];
    for(j=0; j<mlp->classes; j++)
    {
        for(k=0; k<last_hidden_size; k++) 
        {
            mlp->weights[mlp->num_hidden_layers][j][k] -= mlp->learning_rate * deltas[mlp->num_hidden_layers][j] * mlp->outputs[mlp->num_hidden_layers-1][k];
            mlp->weights[mlp->num_hidden_layers][j][k] -= mlp->learning_rate * mlp->l2_lambda * mlp->weights[mlp->num_hidden_layers][j][k];
        }
        mlp->weights[mlp->num_hidden_layers][j][last_hidden_size] -= mlp->learning_rate * deltas[mlp->num_hidden_layers][j];
    }
    
    for(layer=mlp->num_hidden_layers-1; layer>=0; layer--)
    {
        int current_size = mlp->layer_sizes[layer];
        int prev_size = (layer == 0) ? mlp->features : mlp->layer_sizes[layer-1];
        double *prev_output = (layer == 0) ? input : mlp->outputs[layer-1];
        
        for(j=0; j<current_size; j++)
        {
            for(k=0; k<prev_size; k++)
            {
                mlp->weights[layer][j][k] -= mlp->learning_rate * deltas[layer][j] * prev_output[k];
                mlp->weights[layer][j][k] -= mlp->learning_rate * mlp->l2_lambda * mlp->weights[layer][j][k];
            }
            mlp->weights[layer][j][prev_size] -= mlp->learning_rate * deltas[layer][j];
        }
    }
}

void mlp_feedforward(MLP *mlp, double *input)
{
    int i, layer;
    double *current_input = input;
    double *temp_output = malloc_1D(mlp->classes);
    
    if(!temp_output) {
        printf("ERROR: Failed to allocate temp_output in feedforward\n");
        return;
    }
    
    for(layer=0; layer<mlp->num_hidden_layers; layer++)
    {
        int input_size = (layer == 0) ? mlp->features : mlp->layer_sizes[layer-1];
        int output_size = mlp->layer_sizes[layer];
        
        forward_pass(current_input, mlp->weights[layer], output_size, input_size+1, mlp->outputs[layer]);
        
        for(i=0; i<output_size; i++)
        {
            mlp->outputs[layer][i] = relu(mlp->outputs[layer][i]);
        }
        
        current_input = mlp->outputs[layer];
    }
    
    int last_hidden_size = mlp->layer_sizes[mlp->num_hidden_layers - 1];
    forward_pass(current_input, mlp->weights[mlp->num_hidden_layers], mlp->classes, last_hidden_size+1, temp_output);
    
    softmax(temp_output, mlp->outputs[mlp->num_hidden_layers], mlp->classes);
    
    free(temp_output);
}

void mlp_backward(MLP *mlp, double *input, double *class_vector)
{
	int i, j, k, layer;
	double sig_deriv, sum;
	double clip_value = MAX_GRADIENT_CLIP;
	
	double **deltas = (double**)malloc((mlp->num_hidden_layers + 1) * sizeof(double*));
	if(!deltas) return;
	
	for(i=0; i<=mlp->num_hidden_layers; i++)
	{
		if(i < mlp->num_hidden_layers)
			deltas[i] = malloc_1D(mlp->layer_sizes[i]);
		else
			deltas[i] = malloc_1D(mlp->classes);
			
		if(!deltas[i]) {
			for(int cleanup=0; cleanup<i; cleanup++) {
				free(deltas[cleanup]);
			}
			free(deltas);
			return;
		}
	}
	
	for(j=0; j<mlp->classes; j++) 
	{
		deltas[mlp->num_hidden_layers][j] = mlp->outputs[mlp->num_hidden_layers][j] - class_vector[j];
	}
	
	for(layer=mlp->num_hidden_layers-1; layer>=0; layer--)
	{
		int current_size = mlp->layer_sizes[layer];
		int next_size = (layer == mlp->num_hidden_layers-1) ? mlp->classes : mlp->layer_sizes[layer+1];
		
		for(j=0; j<current_size; j++)
		{
			sum = 0.0;
			for(k=0; k<next_size; k++)
			{
				sum += deltas[layer+1][k] * mlp->weights[layer+1][k][j];
			}
			sig_deriv = d_relu(mlp->outputs[layer][j]);
			deltas[layer][j] = sum * sig_deriv;
			
			if(!isfinite(deltas[layer][j])) 
            {
				deltas[layer][j] = 0.0;
			} 
            else 
            {
				if(deltas[layer][j] > clip_value) deltas[layer][j] = clip_value;
				if(deltas[layer][j] < -clip_value) deltas[layer][j] = -clip_value;
			}
		}
	}
	
	weight_update(mlp, input, deltas);
	
	// Clean up deltas array
	for(i=0; i<=mlp->num_hidden_layers; i++) {
		free(deltas[i]);
	}
	free(deltas);
}

void train_mlp(MLP *mlp, double **data, int train_rows, int columns, int epochs)
{
    int i, e, int_class;
    double *input, epoch_error, prev_error = 1000.0;
    double initial_lr = mlp->learning_rate;
    int patience = 0;
    
    if(!mlp || !data || train_rows <= 0 || columns <= 0 || epochs <= 0) {
        printf("ERROR: Invalid parameters for train_mlp\n");
        return;
    }
    
    double *class_vector = malloc_1D(mlp->classes);
    if(!class_vector) 
    {
        printf("ERROR: Failed to allocate class_vector for training\n");
        return;
    }

    printf("Starting training with lr=%.6f...\n", mlp->learning_rate);
    fflush(stdout);
    
    for(e=0; e<epochs; e++)
    {
        epoch_error = 0.0;
        
        printf("Epoch %d: processing %d samples...\n", e, train_rows);
        fflush(stdout);

        for(i=0; i<train_rows; i++)
        {
            input = data[i];
            int_class = (int)input[columns-1];
            
            // Input validation
            if(int_class < 0 || int_class >= mlp->classes) {
                printf("ERROR: Invalid class %d at sample %d\n", int_class, i);
                free(class_vector);
                return;
            }
            
            for(int j=0; j<mlp->classes; j++) 
            {
                class_vector[j] = (j == int_class) ? 1.0 : 0.0;
            }
            
            mlp_feedforward(mlp, input);
            epoch_error += cross_entropy(mlp->outputs[mlp->num_hidden_layers], class_vector, mlp->classes);
            epoch_error += calculate_l2_loss(mlp);
            mlp_backward(mlp, input, class_vector);
            
            // Only show progress every 10% or for small datasets
            if(train_rows <= 10 || (i+1) % (train_rows/10) == 0 || i == train_rows-1) {
                printf("    Sample %d/%d (%.1f%%)    \r", i+1, train_rows, 100.0*(i+1)/train_rows);
                fflush(stdout);
            }
        }

        epoch_error = epoch_error/train_rows;
        
        if(epoch_error > prev_error * 1.1) 
        {
            mlp->learning_rate *= 0.5;
            patience++;
            printf("\n  Epoch %d - Loss: %.6f (INCREASING! Reducing LR to %.6f)\n", e, epoch_error, mlp->learning_rate);
            
            if(patience > 3 || epoch_error > 100.0 || mlp->learning_rate < MIN_LEARNING_RATE) 
            {
                printf("  Training stopped: patience=%d, error=%.2f, lr=%.2e\n", patience, epoch_error, mlp->learning_rate);
                break;
            }
        } 
        else 
        {
            patience = 0;
            printf("\n  Epoch %d - Loss: %.6f\n", e, epoch_error);
        }
        
        prev_error = epoch_error;
        fflush(stdout);
    }
    
    mlp->learning_rate = initial_lr;
    
    free(class_vector);
}

double test_model(MLP *mlp, double **data, int start_row, int end_row, int columns, int classes)
{
    int i, correct = 0, int_class, predicted_class;
    double *input;
    
    if(!mlp || !data || start_row < 0 || end_row <= start_row) {
        printf("ERROR: Invalid parameters for test_model\n");
        return 0.0;
    }
    
    printf("\nTesting...\n");
    
    for(i=start_row; i<end_row; i++) 
    {
        input = data[i];
        int_class = (int)input[columns-1];

        mlp_feedforward(mlp, input);
        predicted_class = arg_max(mlp->outputs[mlp->num_hidden_layers], classes);
        
        // Validate class bounds
        if(int_class < 0 || int_class >= classes) {
            printf("ERROR: Invalid true class %d at sample %d\n", int_class, i);
            continue;
        }
        
        printf("Sample %d: True=%d, Predicted=%d", i-start_row, int_class, predicted_class);
        if(predicted_class == int_class) 
        {
            correct++;
            printf(" V\n");
        }
        else
        {
            // Dynamic output printing for any number of classes
            printf(" X [Output:");
            for(int c=0; c<classes; c++) {
                printf(" %.3f", mlp->outputs[mlp->num_hidden_layers][c]);
            }
            printf("]\n");
        }
    }
    
    return accuracy(correct, end_row, start_row);
}

// Binary format:
// 4 bytes magic: 'MLPB'
// int features
// int classes
// int num_hidden_layers
// int[layer_count] layer_sizes (num_hidden_layers)
// double learning_rate
// double l2_lambda
// For each layer i = 0..num_hidden_layers:
//   int rows, int cols
//   double[rows*cols] weights (row-major)

int save_mlp_binary(MLP *mlp, const char *path)
{
    if(!mlp || !path) return -1;
    FILE *f = fopen(path, "wb");
    if(!f) return -2;

    // Write magic
    char magic[4] = {'M','L','P','B'};
    if(fwrite(magic, 1, 4, f) != 4) { fclose(f); return -3; }

    if(fwrite(&mlp->features, sizeof(int), 1, f) != 1) { fclose(f); return -4; }
    if(fwrite(&mlp->classes, sizeof(int), 1, f) != 1) { fclose(f); return -5; }
    if(fwrite(&mlp->num_hidden_layers, sizeof(int), 1, f) != 1) { fclose(f); return -6; }

    // layer sizes
    if(fwrite(mlp->layer_sizes, sizeof(int), mlp->num_hidden_layers, f) != (size_t)mlp->num_hidden_layers) { fclose(f); return -7; }

    if(fwrite(&mlp->learning_rate, sizeof(double), 1, f) != 1) { fclose(f); return -8; }
    if(fwrite(&mlp->l2_lambda, sizeof(double), 1, f) != 1) { fclose(f); return -9; }

    // Write weights per layer
    for(int layer=0; layer<=mlp->num_hidden_layers; layer++)
    {
        int rows, cols;
        if(layer == 0) {
            rows = mlp->layer_sizes[0];
            cols = mlp->features + 1;
        } else if(layer < mlp->num_hidden_layers) {
            rows = mlp->layer_sizes[layer];
            cols = mlp->layer_sizes[layer-1] + 1;
        } else {
            rows = mlp->classes;
            cols = mlp->layer_sizes[mlp->num_hidden_layers-1] + 1;
        }

        if(fwrite(&rows, sizeof(int), 1, f) != 1) { fclose(f); return -10; }
        if(fwrite(&cols, sizeof(int), 1, f) != 1) { fclose(f); return -11; }

        // weights is double** contiguous for each row
        for(int i=0; i<rows; i++) {
            if(fwrite(mlp->weights[layer][i], sizeof(double), cols, f) != (size_t)cols) { fclose(f); return -12; }
        }
    }

    fclose(f);
    return 0;
}

MLP* load_mlp_binary(const char *path)
{
    if(!path) return NULL;
    FILE *f = fopen(path, "rb");
    if(!f) return NULL;

    char magic[4];
    if(fread(magic, 1, 4, f) != 4) { fclose(f); return NULL; }
    if(!(magic[0]=='M' && magic[1]=='L' && magic[2]=='P' && magic[3]=='B')) { fclose(f); return NULL; }

    MLP *mlp = (MLP*)malloc(sizeof(MLP));
    if(!mlp) { fclose(f); return NULL; }
    memset(mlp, 0, sizeof(MLP));

    if(fread(&mlp->features, sizeof(int), 1, f) != 1) { free(mlp); fclose(f); return NULL; }
    if(fread(&mlp->classes, sizeof(int), 1, f) != 1) { free(mlp); fclose(f); return NULL; }
    if(fread(&mlp->num_hidden_layers, sizeof(int), 1, f) != 1) { free(mlp); fclose(f); return NULL; }

    mlp->layer_sizes = malloc_1D_int(mlp->num_hidden_layers);
    if(!mlp->layer_sizes) { free(mlp); fclose(f); return NULL; }
    if(fread(mlp->layer_sizes, sizeof(int), mlp->num_hidden_layers, f) != (size_t)mlp->num_hidden_layers) { free(mlp->layer_sizes); free(mlp); fclose(f); return NULL; }

    if(fread(&mlp->learning_rate, sizeof(double), 1, f) != 1) { free(mlp->layer_sizes); free(mlp); fclose(f); return NULL; }
    if(fread(&mlp->l2_lambda, sizeof(double), 1, f) != 1) { free(mlp->layer_sizes); free(mlp); fclose(f); return NULL; }

    // allocate containers
    mlp->weights = malloc_3D_partial(mlp->num_hidden_layers + 1);
    mlp->outputs = malloc_2D_partial(mlp->num_hidden_layers + 1);
    if(!mlp->weights || !mlp->outputs) {
        if(mlp->weights) free(mlp->weights);
        if(mlp->outputs) free(mlp->outputs);
        free(mlp->layer_sizes);
        free(mlp);
        fclose(f);
        return NULL;
    }

    for(int layer=0; layer<=mlp->num_hidden_layers; layer++)
    {
        int rows, cols;
        if(fread(&rows, sizeof(int), 1, f) != 1) { free_mlp(mlp); fclose(f); return NULL; }
        if(fread(&cols, sizeof(int), 1, f) != 1) { free_mlp(mlp); fclose(f); return NULL; }

        // allocate matrix contiguous rows x cols
        mlp->weights[layer] = malloc_2D_complete(rows, cols);
        if(!mlp->weights[layer]) { free_mlp(mlp); fclose(f); return NULL; }

        for(int i=0; i<rows; i++) {
            if(fread(mlp->weights[layer][i], sizeof(double), cols, f) != (size_t)cols) { free_mlp(mlp); fclose(f); return NULL; }
        }

        // allocate outputs for this layer
        if(layer < mlp->num_hidden_layers)
            mlp->outputs[layer] = malloc_1D(mlp->layer_sizes[layer]);
        else
            mlp->outputs[layer] = malloc_1D(mlp->classes);
        if(!mlp->outputs[layer]) { free_mlp(mlp); fclose(f); return NULL; }
    }

    fclose(f);
    return mlp;
}

#endif /* TRN_FNCS_H_ */
