/*
 * MemoryCell.cpp
 *
 *  Created on: Jul 18, 2016
 *      Author: trabucco
 */

#include "MemoryBlock.h"

long long int MemoryBlock::n = 0;

MemoryBlock::MemoryBlock(int cl, int cn) :
		bias(3),
		cells(cl),
		inputFeedbackWeight(cl),
		inputStateWeight(cl),
		forgetFeedbackWeight(cl),
		forgetStateWeight(cl),
		outputFeedbackWeight(cl),
		outputStateWeight(cl),
		inputDataWeight(cn),
		forgetDataWeight(cn),
		outputDataWeight(cn)
	{
	// TODO Auto-generated constructor stub
	nConnections = cn;
	nCells = cl;

	bias[0] = 0;
	bias[1] = 0;
	bias[2] = 0;

	default_random_engine g(time(0) + (n++));
	normal_distribution<double> d(0, 1);

	for (int i = 0; i < nCells; i++) {
		cells[i] = (MemoryCell(nConnections));
		inputFeedbackWeight[i] = (d(g));
		inputStateWeight[i] = (d(g));
		forgetFeedbackWeight[i] = (d(g));
		forgetStateWeight[i] = (d(g));
		outputFeedbackWeight[i] = (d(g));
		outputStateWeight[i] = (d(g));
	}

	for (int i = 0; i < nConnections; i++) {
		inputDataWeight[i] = (d(g));
		forgetDataWeight[i] = (d(g));
		outputDataWeight[i] = (d(g));
	}
}

MemoryBlock::~MemoryBlock() {
	// TODO Auto-generated destructor stub
}


double MemoryBlock::inputGate(double data) {
	double in = sigmoid(data);
	input.push_back(in);
	inputPrime.push_back(sigmoidPrime(data));
	return in;
}

double MemoryBlock::forgetGate(double data) {
	double forg = sigmoid(data);
	forget.push_back(forg);
	forgetPrime.push_back(sigmoidPrime(data));
	return forg;
}

double MemoryBlock::outputGate(double data) {
	double out = sigmoid(data);
	output.push_back(out);
	outputPrime.push_back(sigmoidPrime(data));
	return out;
}

vector<double> MemoryBlock::forward(vector<double> input, int t) {
	vector<double> cellSum(nCells);
	double inputSum = bias[0];
	double forgetSum = bias[1];
	double outputSum = bias[2];

	impulse.push_back(input);

	for (int i = 0; i < nCells; i++) {
		inputSum += (inputFeedbackWeight[i] * cells[i].feedback[t]) +
				(inputStateWeight[i] * cells[i].state[t]);
		forgetSum += (forgetFeedbackWeight[i] * cells[i].feedback[t]) +
				(forgetStateWeight[i] * cells[i].state[t]);
		outputSum += (outputFeedbackWeight[i] * cells[i].feedback[t]) +
				(outputStateWeight[i] * cells[i].state[t]);
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

	double forgetActivation = forgetGate(forgetSum);
	double inputActivation = inputGate(inputSum);
	double outputActivation = outputGate(outputSum);

	// compute input into memory
	vector<double> output(nCells);
	for (int i = 0; i < nCells; i++) {
		cells[i].previousState.push_back(cells[i].state[t]);
		cells[i].state.push_back(cells[i].state[t] * forgetActivation);
		cells[i].state[t + 1] += cells[i].activateIn(cellSum[i]) * inputActivation;

		// compute output of memory cell
		cells[i].previousFeedback.push_back(cells[i].feedback[t]);
		cells[i].feedback.push_back(cells[i].activateOut(cells[i].state[t + 1]) * outputActivation);
		output[i] = (cells[i].feedback[t + 1]);
	} return output;
}

// errorprime must be a vector with length of number of cells
vector<double>  MemoryBlock::backward(vector<double>  errorPrime, double learningRate, int t, int length) {
	int p = (length - 1 - t);
	vector<double> eta(nCells),
			inputDataPartialSum(nConnections, 0),
			forgetDataPartialSum(nConnections, 0);
	double blockSum = 0,
			inputFeedbackPartialSum = 0,
			inputStatePartialSum = 0,
			forgetFeedbackPartialSum = 0,
			forgetStatePartialSum = 0;

	for (int i = 0; i < nCells; i++) {
		double recurrentError = errorPrime[i];

		blockSum += cells[i].activationOut[t] * recurrentError;
		eta[i] = (output[t] * cells[i].activationOutPrime[t] * (recurrentError));
		cells[i].internalError = eta[i];
	} blockSum *= outputPrime[t];

	for (int i = 0; i < nConnections; i++) {
		outputDataWeight[i] -= learningRate * blockSum * impulse[t][i];	// invalid read of size 8
	}

	// calculate the updates, and update the cell weights
	for (int i = 0; i < nCells; i++) {
		outputFeedbackWeight[i] -= learningRate * blockSum * cells[i].feedback[t + 1];
		outputStateWeight[i] -= learningRate * blockSum * cells[i].state[t + 1];
		cells[i].cellDataPartial.push_back(vector<double>(nConnections));
		cells[i].forgetDataPartial.push_back(vector<double>(nConnections));
		cells[i].inputDataPartial.push_back(vector<double>(nConnections));
		for (int j = 0; j < nConnections; j++) {
			cells[i].cellDataPartial[p + 1][j] = cells[i].cellDataPartial[p][j] * forget[t] + cells[i].activationInPrime[t] * input[t] * impulse[t][j];
			cells[i].cellDataWeight[j] -= learningRate * eta[i] * cells[i].cellDataPartial[p + 1][j];
			cells[i].forgetDataPartial[p + 1][j] = cells[i].forgetDataPartial[p][j] * forget[t] + cells[i].previousState[t] * forgetPrime[t] * impulse[t][j];	// invalid read of size 8
			cells[i].inputDataPartial[p + 1][j] = cells[i].inputDataPartial[p][j] * forget[t] + cells[i].activationIn[t] * inputPrime[t] * impulse[t][j];	// invalid read of size 8
			forgetDataPartialSum[j] += cells[i].forgetDataPartial[p + 1][j] * eta[i];
			inputDataPartialSum[j] += cells[i].inputDataPartial[p + 1][j] * eta[i];
		}

		cells[i].cellFeedbackPartial.push_back(cells[i].cellFeedbackPartial[p] * forget[t] + cells[i].activationInPrime[t] * input[t] * cells[i].previousFeedback[t + 1]);
		cells[i].cellFeedbackWeight -= learningRate * eta[i] * cells[i].cellFeedbackPartial[p + 1];

		cells[i].forgetFeedbackPartial.push_back(cells[i].forgetFeedbackPartial[p] * forget[t] + cells[i].activationIn[t] * forgetPrime[t] * cells[i].previousFeedback[t + 1]);
		cells[i].forgetStatePartial.push_back(cells[i].forgetStatePartial[p] * forget[t] + cells[i].activationIn[t] * forgetPrime[t] * cells[i].previousState[t + 1]);
		forgetFeedbackPartialSum += eta[i] * cells[i].forgetFeedbackPartial[p + 1];
		forgetStatePartialSum += eta[i] * cells[i].forgetStatePartial[p + 1];

		cells[i].inputFeedbackPartial.push_back(cells[i].inputFeedbackPartial[p] * forget[t] + cells[i].activationIn[t] * inputPrime[t] * cells[i].previousFeedback[t + 1]);
		cells[i].inputStatePartial.push_back(cells[i].inputStatePartial[p] * forget[t] + cells[i].activationIn[t] * inputPrime[t] * cells[i].previousState[t + 1]);
		inputFeedbackPartialSum += eta[i] * cells[i].inputFeedbackPartial[p + 1];
		inputStatePartialSum += eta[i] * cells[i].inputStatePartial[p + 1];
	}

	// update the input, output, and forget weights
	for (int j = 0; j < nConnections; j++) {
		forgetDataWeight[j] -= learningRate * forgetDataPartialSum[j];	// invalid read of size 8
		inputDataWeight[j] -= learningRate * inputDataPartialSum[j];	// invalid read of size 8
	}

	for (int i = 0; i < nCells; i++) {
		inputFeedbackWeight[i] -= learningRate * inputFeedbackPartialSum;
		inputStateWeight[i] -= learningRate * inputFeedbackPartialSum;
		forgetFeedbackWeight[i] -= learningRate * forgetFeedbackPartialSum;
		forgetStateWeight[i] -= learningRate * forgetStatePartialSum;
	}

	vector<double> temp(nConnections);
	for (int i = 0; i < nConnections; i++) {
		temp[i] = (0.0);
		for (int j = 0; j < nCells; j++) {
			temp[i] += (cells[j].internalError * cells[j].cellDataWeight[i]) +
					(cells[j].internalError * forgetDataWeight[i]) +
					(cells[j].internalError * inputDataWeight[i]) +
					(blockSum * outputDataWeight[i]);
		}
	} return temp;
}

void MemoryBlock::clear() {
	for (int i = 0; i < nCells; i++) {
		cells[i].clear();
	} impulse.clear();
	input.clear();
	inputPrime.clear();
	forget.clear();
	forgetPrime.clear();
	output.clear();
	outputPrime.clear();
}
