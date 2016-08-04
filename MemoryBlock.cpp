/*
 * MemoryCell.cpp
 *
 *  Created on: Jul 18, 2016
 *      Author: trabucco
 */

#include "MemoryBlock.h"

long long int MemoryBlock::n = 0;

MemoryBlock::MemoryBlock(int cl, int cn) {
	// TODO Auto-generated constructor stub
	nConnections = cn;
	nCells = cl;
	input = 0; inputPrime = 0;
	forget = 0; forgetPrime = 0;
	output = 0; outputPrime = 0;

	default_random_engine g(time(0) + (n++));
	normal_distribution<double> d(0, 1);

	bias = (double *)calloc(3, sizeof(double));
	cells = (MemoryCell *)malloc(sizeof(MemoryCell) * nCells);
	inputFeedbackWeight = (double *)malloc(sizeof(double) * nCells);
	inputStateWeight = (double *)malloc(sizeof(double) * nCells);
	forgetFeedbackWeight = (double *)malloc(sizeof(double) * nCells);
	forgetStateWeight = (double *)malloc(sizeof(double) * nCells);
	outputFeedbackWeight = (double *)malloc(sizeof(double) * nCells);
	outputStateWeight = (double *)malloc(sizeof(double) * nCells);

	for (int i = 0; i < nCells; i++) {
		cells[i] = (MemoryCell(nConnections));
		inputFeedbackWeight[i] = (d(g));
		inputStateWeight[i] = (d(g));
		forgetFeedbackWeight[i] = (d(g));
		forgetStateWeight[i] = (d(g));
		outputFeedbackWeight[i] = (d(g));
		outputStateWeight[i] = (d(g));
	}

	impulse = (double *)malloc(sizeof(double) * nConnections);
	inputDataWeight = (double *)malloc(sizeof(double) * nConnections);
	forgetDataWeight = (double *)malloc(sizeof(double) * nConnections);
	outputDataWeight = (double *)malloc(sizeof(double) * nConnections);

	for (int i = 0; i < nConnections; i++) {
		impulse[i] = (0);
		inputDataWeight[i] = (d(g));
		forgetDataWeight[i] = (d(g));
		outputDataWeight[i] = (d(g));
	}
}

MemoryBlock::~MemoryBlock() {
	// TODO Auto-generated destructor stub
	free(bias);
	free(cells);
	free(inputFeedbackWeight);
	free(inputStateWeight);
	free(forgetFeedbackWeight);
	free(forgetStateWeight);
	free(outputFeedbackWeight);
	free(outputStateWeight);
	free(impulse);
	free(inputDataWeight);
	free(forgetDataWeight);
	free(outputDataWeight);
}


double MemoryBlock::inputGate(double data) {
	input = sigmoid(data);
	inputPrime = sigmoidPrime(data);
	return input;
}

double MemoryBlock::forgetGate(double data) {
	forget = sigmoid(data);
	forgetPrime = sigmoidPrime(data);
	return forget;
}

double MemoryBlock::outputGate(double data) {
	output = sigmoid(data);
	outputPrime = sigmoidPrime(data);
	return output;
}

double *MemoryBlock::forward(double *input) {
	double *cellSum = (double *)calloc(nCells, sizeof(double));
	double inputSum = bias[0];
	double forgetSum = bias[1];
	double outputSum = bias[2];

	memcpy(impulse, input, (sizeof(double) * nConnections));

	for (int i = 0; i < nCells; i++) {
		inputSum += (inputFeedbackWeight[i] * cells[i].feedback) +
				(inputStateWeight[i] * cells[i].state);
		forgetSum += (forgetFeedbackWeight[i] * cells[i].feedback) +
				(forgetStateWeight[i] * cells[i].state);
		outputSum += (outputFeedbackWeight[i] * cells[i].feedback) +
				(outputStateWeight[i] * cells[i].state);
	}

	// find the weighted sum of all input
	for (int i = 0; i < nConnections; i++) {
		for (unsigned int j = 0; j < nCells; j++) {
			cellSum[j] += input[i] * cells[j].cellDataWeight[i];
		}
		inputSum += input[i] * inputDataWeight[i];
		forgetSum += input[i] * forgetDataWeight[i];
		outputSum += input[i] * outputDataWeight[i];
	}

	// compute input into memory
	double *output = (double *)malloc(sizeof(double) * nCells);
	for (int i = 0; i < nCells; i++) {
		cells[i].previousState = cells[i].state;
		cells[i].state *= forgetGate(forgetSum);
		cells[i].state += cells[i].activateIn(cellSum[i]) * inputGate(inputSum);

		// compute output of memory cell
		cells[i].previousFeedback = cells[i].feedback;
		cells[i].feedback = cells[i].activateOut(cells[i].state) * outputGate(outputSum);
		output[i] = (cells[i].feedback);
	}

	free(cellSum);

	return output;
}

// errorprime must be a vector with length of number of cells
double *MemoryBlock::backward(double *errorPrime, double learningRate) {
	double *eta = (double *)malloc(sizeof(double) * nCells),
			*inputDataPartialSum = (double *)calloc(nConnections, sizeof(double)),
			*forgetDataPartialSum =  (double *)calloc(nConnections, sizeof(double));
	double blockSum = 0,
			inputFeedbackPartialSum = 0,
			inputStatePartialSum = 0,
			forgetFeedbackPartialSum = 0,
			forgetStatePartialSum = 0;

	for (int i = 0; i < nCells; i++) {
		blockSum += cells[i].activationOut * errorPrime[i];
		eta[i] = (output * cells[i].activationOutPrime * errorPrime[i]);
		outputFeedbackWeight[i] -= learningRate * blockSum * outputPrime * cells[i].feedback;
		outputStateWeight[i] -= learningRate * blockSum * outputPrime * cells[i].state;
	}

	for (int i = 0; i < nConnections; i++) {
		outputDataWeight[i] -= learningRate * blockSum * outputPrime * impulse[i];	// invalid read of size 8
	}

	// calculate the updates, and update the cell weights
	for (int i = 0; i < nCells; i++) {
		for (int j = 0; j < nConnections; j++) {
			cells[i].cellDataPartial[j] = cells[i].cellDataPartial[j] * forget + cells[i].activationInPrime * input * impulse[j];
			cells[i].cellDataWeight[j] -= learningRate * eta[i] * cells[i].cellDataPartial[j];
			cells[i].forgetDataPartial[j] = cells[i].forgetDataPartial[j] * forget + cells[i].previousState * forgetPrime * impulse[j];	// invalid read of size 8
			cells[i].inputDataPartial[j] = cells[i].inputDataPartial[j] * forget + cells[i].activationIn * inputPrime * impulse[j];	// invalid read of size 8
			forgetDataPartialSum[j] += cells[i].forgetDataPartial[j] * eta[i];
			inputDataPartialSum[j] += cells[i].inputDataPartial[j] * eta[i];
		}

		cells[i].cellFeedbackPartial = cells[i].cellFeedbackPartial * forget + cells[i].activationInPrime * input * cells[i].previousFeedback;
		cells[i].cellFeedbackWeight -= learningRate * eta[i] * cells[i].cellFeedbackPartial;

		cells[i].forgetFeedbackPartial = cells[i].forgetFeedbackPartial * forget + cells[i].previousState * forgetPrime * cells[i].previousFeedback;
		cells[i].forgetStatePartial = cells[i].forgetStatePartial * forget + cells[i].previousState * forgetPrime * cells[i].previousState;
		forgetFeedbackPartialSum += eta[i] *cells[i].forgetFeedbackPartial;
		forgetStatePartialSum += eta[i] *cells[i].forgetStatePartial;

		cells[i].inputFeedbackPartial = cells[i].inputFeedbackPartial * forget + cells[i].activationIn * inputPrime * cells[i].previousFeedback;
		cells[i].inputStatePartial = cells[i].inputStatePartial * forget + cells[i].activationIn * inputPrime * cells[i].previousState;
		inputFeedbackPartialSum += eta[i] *cells[i].inputFeedbackPartial;
		inputStatePartialSum += eta[i] *cells[i].inputStatePartial;
	}

	// update the input, output, and forget weights
	for (int i = 0; i < nCells; i++) {
		for (int j = 0; j < nConnections; j++) {
			forgetDataWeight[j] -= learningRate * forgetDataPartialSum[j];	// invalid read of size 8
			inputDataWeight[j] -= learningRate * inputDataPartialSum[j];	// invalid read of size 8
		}
		inputFeedbackWeight[i] -= learningRate * inputFeedbackPartialSum;
		inputStateWeight[i] -= learningRate * inputFeedbackPartialSum;
		forgetFeedbackWeight[i] -= learningRate * forgetFeedbackPartialSum;
		forgetStateWeight[i] -= learningRate * forgetStatePartialSum;
	}

	double *temp = (double *)malloc(sizeof(double) * nConnections);
	for (int i = 0; i < nConnections; i++) {
		temp[i] = (0.0);
	}

	free(eta);
	free(inputDataPartialSum);
	free(forgetDataPartialSum);

	return temp;
}

