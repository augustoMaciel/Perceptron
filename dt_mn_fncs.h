/*
 * dt_mn_fncs.h
 *
 *  Created on: 11 de jun. de 2025
 *      Author: Augusto Lipinski Fernandes Maciel
 */

#ifndef DT_MN_FNCS_H_
#define DT_MN_FNCS_H_

void merge(double *arr, int left, int mid, int right)
{
    int i, j, k;
    int n1 = mid - left + 1;
    int n2 = right - mid;
    double *L = (double*)malloc(n1 * sizeof(double));
    double *R = (double*)malloc(n2 * sizeof(double));
    
    for(i=0; i<n1; i++)
        L[i] = arr[left + i];
    for(j=0; j<n2; j++)
        R[j] = arr[mid + 1 + j];
    
    i = 0;
    j = 0;
    k = left;
    
    while(i < n1 && j < n2)
    {
        if(L[i] <= R[j])
        {
            arr[k] = L[i];
            i++;
        }
        else
        {
            arr[k] = R[j];
            j++;
        }
        k++;
    }
    
    while(i < n1)
    {
        arr[k] = L[i];
        i++;
        k++;
    }
    
    while(j < n2)
    {
        arr[k] = R[j];
        j++;
        k++;
    }
    
    free(L);
    free(R);
}

void merge_sort(double *arr, int left, int right)
{
    if(left < right)
    {
        int mid = left + (right - left) / 2;
        merge_sort(arr, left, mid);
        merge_sort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

int get_rows_number (char *path)
{
	int n = 0;
	FILE *file = fopen(path, "r");
	char buffer[256];

	while(fgets(buffer, sizeof(buffer), file))
	{
		n++;
	}

	fclose(file);
	return n;
}

int get_columns_number (char *path, const char *delimiter, int buffer_size)
{
	int n = 0;
	FILE *file = fopen(path, "r");
	char *token, buffer[buffer_size];

	fgets(buffer, sizeof(buffer), file);
	token = strtok(buffer, delimiter);
	while(token!=NULL)
	{
		n++;
		token = strtok(NULL, delimiter); 
	}
	
	fclose(file);
	return n;
}

int get_classes_number (char *path, const char *delimiter, int buffer_size, int rows)
{
	int n = 0, i = 0;
	double *classes;
	FILE *file = fopen(path, "r");
	char *token, *last_token, buffer[buffer_size];

	if(!file) {
		printf("ERROR: no such file as %s.\n", path);
		fflush(stdout);
		return 0;
	}

	classes = (double*)malloc(rows * sizeof(double));
	if(!classes) {
		fclose(file);
		return 0;
	}

	i = 0;
	while(fgets(buffer, buffer_size, file) != NULL && i < rows)
	{
		buffer[strcspn(buffer, "\n")] = 0;

		// Tokenize the line and keep the last token
		last_token = NULL;
		token = strtok(buffer, delimiter);
		while(token != NULL)
		{
			last_token = token;
			token = strtok(NULL, delimiter);
		}

		if(last_token != NULL)
		{
			classes[i++] = atof(last_token);
		}
	}

	if(i == 0)
	{
		free(classes);
		fclose(file);
		return 0;
	}

	// sort only the collected entries
	merge_sort(classes, 0, i-1);

	n = 1;
	for(int k = 1; k < i; k++)
	{
		if(classes[k] != classes[k-1]) n++;
	}

	free(classes);
	fclose(file);
	return n;
}

int* get_meta_data_from_csv (char *path, const char *delimiter, int buffer_size)
{
	int i=0, j=0, k, rows, columns, classes, features, *meta_data;
	double num;
	FILE *file = fopen(path, "r");
	char *token, buffer[buffer_size];

	if(!file)
	{
		printf("ERROR: no such file as %s.\n", path);
		fflush(stdout);
		return NULL;
	}
	
	rows = get_rows_number(path);
	columns = get_columns_number(path, delimiter, buffer_size);
	classes = get_classes_number(path, delimiter, buffer_size, rows);
	features = columns-1;
	
	meta_data = (int*)malloc(4*sizeof(int));
	if(!meta_data) {
		fclose(file);
		return NULL;
	}
	meta_data[0] = rows;
	meta_data[1] = columns;
	meta_data[2] = classes;
	meta_data[3] = features;
	
	fclose(file);
	return meta_data;
}

double** read_from_csv (char *path, int buffer_size, int rows, int columns)
{
	int i = 0, j = 0, k;
	double num;
	FILE *file = fopen(path, "r");
	char *token, buffer[buffer_size];
	const char *delimiter = ",";
	int progress_step = rows / 10;

	if(!file)
	{
		printf("ERROR: no such file as %s.\n", path);
		fflush(stdout);
		return NULL;
	}
	
	size_t total_size = (size_t)rows * sizeof(double*);
	if(total_size / sizeof(double*) != (size_t)rows) 
	{
		printf("ERROR: Memory allocation would overflow\n");
		fclose(file);
		return NULL;
	}
	
	double** matrix = (double**)malloc(total_size);
	if(!matrix) {
		printf("ERROR: Failed to allocate matrix row pointers (%zu bytes)\n", total_size);
		fclose(file);
		return NULL;
	}
	for(k=0; k<rows; k++)
	{
		size_t row_size = (size_t)columns * sizeof(double);
		if(row_size / sizeof(double) != (size_t)columns) 
		{
			printf("ERROR: Row allocation would overflow at row %d\n", k);
			for(int cleanup=0; cleanup<k; cleanup++) free(matrix[cleanup]);
			free(matrix);
			fclose(file);
			return NULL;
		}
		
		matrix[k] = (double*)malloc(row_size);
		if(!matrix[k]) {
			printf("ERROR: Failed to allocate row %d (%zu bytes)\n", k, row_size);
			for(int cleanup=0; cleanup<k; cleanup++) free(matrix[cleanup]);
			free(matrix);
			fclose(file);
			return NULL;
		}
		
		if(progress_step > 0 && (k % progress_step == 0 || k == rows-1)) 
		{
			printf("  Allocating memory: %d/%d rows (%.1f%%)\r", k+1, rows, 100.0*(k+1)/rows);
			fflush(stdout);
		}
	}

	printf("\n");
	while(fgets(buffer, buffer_size, file)!=NULL)
	{
		buffer[strcspn(buffer, "\n")] = 0;
		token = strtok(buffer, delimiter);
		while(token!=NULL)
		{
			num = atof(token);
			matrix[i][j] = num;
			token = strtok(NULL, delimiter);
			j++;
		}
		j=0;
		i++;
		
		printf("  Reading CSV: %d/%d rows (%.1f%%)\r", i+1, rows, 100.0*(i+1)/rows);
		fflush(stdout);
	}
	
	printf("\n");
	size_t total_memory = (size_t)rows * columns * sizeof(double) + rows * sizeof(double*);
	printf("Dataset loaded successfully. Memory used: %.2f MB\n", total_memory / (1024.0 * 1024.0));
	fclose(file);
	return matrix;
}

void print_matrix (double **matrix, int rows, int columns, char *who)
{
	int i, j;
	
	printf("Showing values of %s:\n", who);
	for(i=0; i<rows; i++)
	{
		printf("ROW[%d]: ", i);
		fflush(stdout);
		for(j=0; j<columns; j++)
		{
			printf("%.3f ", matrix[i][j]);
			fflush(stdout);
		}
		printf("\n");
		fflush(stdout);
	}
}

void print_meta_data (int *vector)
{
	printf("Dataset info - Rows: %d, Columns: %d, Features: %d, Classes: %d\n", vector[0], vector[1], vector[3], vector[2]);
}

double* malloc_1D (int rows)
{
	double *vector = (double*)malloc(rows*sizeof(double));
	return vector;
}

int* malloc_1D_int (int rows)
{
	int *vector = (int*)malloc(rows*sizeof(int));
	return vector;
}

double** malloc_2D_complete (int rows, int columns)
{
	int i;
	
	double *data = (double*)malloc(rows * columns * sizeof(double));
	if(!data) return NULL;
	
	double **matrix = (double**)malloc(rows*sizeof(double*));
	if(!matrix) 
	{
		free(data);
		return NULL;
	}
	
	for(i=0; i<rows; i++)
	{
		matrix[i] = data + i * columns;
	}
	return matrix;
}

double** malloc_2D_partial (int size)
{
	double **matrix = (double**)malloc((size) * sizeof(double*));
	return matrix;
}

double*** malloc_3D_partial (int size)
{
	double ***matrix = (double***)malloc((size) * sizeof(double**));
	return matrix;
}

void shuffle(double **data, int rows) 
{
    int i, j;
    double *aux;
    for(i=rows-1; i>0; i--) 
    {
        j = rand()%(i+1);
        aux = data[i];
        data[i] = data[j];
        data[j] = aux;
    }
}

int arg_max(double *vector, int size) 
{
    int i, max_index = 0;
    for(i=1; i<size; i++) 
    {
        if(vector[i] > vector[max_index]) 
        {
            max_index = i;
        }
    }
    return max_index;
}

void normalize_dataset(double **data, int rows, int columns)
{
	int i, j;
	double min, max;
	
    for(j=0; j<columns-1; j++)
	{
        min = data[0][j];
        max = data[0][j];

        for(i=1; i<rows; i++)
		{
            if(data[i][j]<min)
			{
				min = data[i][j];
			}
            if(data[i][j]>max)
			{
				max = data[i][j];
			}
		}

        for(i=0; i<rows; i++)
		{
            data[i][j] = (data[i][j]-min)/(max-min);
        }
    }
}

// Normalize a single sample using min-max values computed from a dataset.
// `features` is the number of feature columns in `data` and the length of `sample`.
// This uses the same per-column min-max formula as `normalize_dataset`.
void normalize_sample_using_dataset(double *sample, double **data, int rows, int features)
{
	if(!sample || !data || rows <= 0 || features <= 0) return;
	int i, j;
	double min, max;

	for(j=0; j<features; j++)
	{
		min = data[0][j];
		max = data[0][j];
		for(i=1; i<rows; i++)
		{
			if(data[i][j] < min) min = data[i][j];
			if(data[i][j] > max) max = data[i][j];
		}

		if(max != min)
		{
			sample[j] = (sample[j] - min) / (max - min);
		}
		else
		{
			sample[j] = 0.0;
		}
	}
}

void free_matrix (double **matrix, int rows)
{
	int i;
	
	if(!matrix) return;
	
	for(i=0; i<rows; i++)
	{
		if(matrix[i]) free(matrix[i]);
	}
	free(matrix);
}

void free_matrix_contiguous (double **matrix, int rows)
{
	if(!matrix) return;
	
	if(matrix[0]) 
	{
		free(matrix[0]);
	}
	
	free(matrix);
}

// Build a 1-row matrix from an array of features provided at runtime.
// The returned matrix has 1 row and `features_count` columns.
// This matrix contains only features (no class/label column) and is
// suitable for inference. Caller is responsible for freeing the matrix
// using `free_matrix(matrix, 1)` when done.
double** build_matrix_from_features(const double *features, int features_count)
{
	if(!features || features_count <= 0) return NULL;

	int rows = 1;
	int columns = features_count; // only features, no class column

	double **matrix = malloc_2D_partial(rows);
	if(!matrix) return NULL;

	matrix[0] = (double*)malloc(columns * sizeof(double));
	if(!matrix[0]) {
		free(matrix);
		return NULL;
	}

	for(int j=0; j<features_count; j++) {
		matrix[0][j] = features[j];
	}

	return matrix;
}

#endif /* DT_MN_FNCS_H_ */
