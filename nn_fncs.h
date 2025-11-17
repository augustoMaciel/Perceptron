/*
 * nn_fncs.h
 *
 *  Created on: 13 de nov. de 2025
 *      Author: Augusto Lipinski Fernandes Maciel
 */

#ifndef NN_FUNCS_H_
#define NN_FUNCS_H_

double sigmoid(double x)
{
    return 1.0/(1.0+exp(-x));
}

double d_sigmoid(double sig)
{
    return sig*(1.0-sig);
}

void softmax(double *input, double *output, int size)
{
    int i;
    double sum = 0.0;
    double max = input[0];
    
    for(i=1; i<size; i++)
    {
        if(input[i] > max)
        {
            max = input[i];
        }
    }
    
    for(i=0; i<size; i++)
    {
        output[i] = exp(input[i] - max);
        sum += output[i];
    }
    
    for(i=0; i<size; i++)
    {
        output[i] /= sum;
    }
}

#endif /* NN_FUNCS_H_ */