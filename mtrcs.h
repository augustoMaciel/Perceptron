/*
 * mtrcs.h
 *
 *  Created on: 13 de nov. de 2025
 *      Author: Augusto Lipinski Fernandes Maciel
 */

#ifndef MTRCS_H_
#define MTRCS_H_

double mean_squared_err(double *predicted, double *expected, int size) 
{
	int i;
	double err = 0;
	
    for(i=0; i<size; i++) 
	{
        err += pow(expected[i]-predicted[i], 2);
    }
    return err/size;
}

double accuracy (int correct, int end_row, int start_row)
{
    return 100.0*correct/(end_row-start_row);
}

double cross_entropy(double *predicted, double *expected, int size)
{
    int i;
    double loss = 0.0;
    double epsilon = 1e-15;
    
    for(i=0; i<size; i++)
    {
        double pred = predicted[i];
        if(pred < epsilon) pred = epsilon;
        if(pred > 1.0 - epsilon) pred = 1.0 - epsilon;
        
        loss -= expected[i] * log(pred);
    }
    
    return loss;
}

#endif /* MTRCS_H_ */