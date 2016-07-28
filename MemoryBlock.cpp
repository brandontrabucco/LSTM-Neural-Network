/*
 * MemoryCell.cpp
 *
 *  Created on: Jul 18, 2016
 *      Author: trabucco
 */

#include "MemoryBlock.h"

long long int MemoryBlock::n = 0;

MemoryBlock::MemoryBlock(int nCells, int nConnections) {
	// TODO Auto-generated constructor stub
	connections = nConnections;
	input = 0; inputPrime = 0;
	forget = 0; forgetPrime = 0;
	output = 0; outputPrime = 0;

	default_random_engine g(time(0) + (n++));
	normal_distribution<double> d(0, 1);

	bias.push_back(0);
	bias.push_back(0);
	bias.push_back(0);

	for (int i = 0; i < nCells; i++) {
		cells.push_back(MemoryCell(nConnections));
		inputFeedbackWeight.push_back(d(g));
		inputStateWeight.push_back(d(g));
		forgetFeedbackWeight.push_back(d(g));
		forgetStateWeight.push_back(d(g));
		outputFeedbackWeight.push_back(d(g));
		outputStateWeight.push_back(d(g));
	}

	for (int i = 0; i < nConnections; i++) {
		impulse.push_back(0);
		inputDataWeight.push_back(d(g));	// invalid memory
		forgetDataWeight.push_back(d(g));
		outputDataWeight.push_back(d(g));
	}
}

MemoryBlock::~MemoryBlock() {
	// TODO Auto-generated destructor stub
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

vector<double> MemoryBlock::forward(vector<double> input) {
	vector<double> cellSum = vector<double>(cells.size(), 0.0);
	double inputSum = bias[0];
	double forgetSum = bias[1];
	double outputSum = bias[2];

	impulse = input;

	for (unsigned int i = 0; i < cells.size(); i++) {
		inputSum += (inputFeedbackWeight[i] * cells[i].feedback) +
				(inputStateWeight[i] * cells[i].state);
		forgetSum += (forgetFeedbackWeight[i] * cells[i].feedback) +
				(forgetStateWeight[i] * cells[i].state);
		outputSum += (outputFeedbackWeight[i] * cells[i].feedback) +
				(outputStateWeight[i] * cells[i].state);
	}

	// find the weighted sum of all input
	for (unsigned int i = 0; i < input.size(); i++) {
		for (unsigned int j = 0; j < cells.size(); j++) {
			cellSum[j] += input[i] * cells[j].cellDataWeight[i];
		}
		inputSum += input[i] * inputDataWeight[i];
		forgetSum += input[i] * forgetDataWeight[i];
		outputSum += input[i] * outputDataWeight[i];
	}

	// compute input into memory
	vector<double> output;
	for (unsigned int i = 0; i < cells.size(); i++) {
		cells[i].previousState = cells[i].state;
		cells[i].state *= forgetGate(forgetSum);
		cells[i].state += cells[i].activateIn(cellSum[i]) * inputGate(inputSum);

		// compute output of memory cell
		cells[i].previousFeedback = cells[i].feedback;
		cells[i].feedback = cells[i].activateOut(cells[i].state) * outputGate(outputSum);
		output.push_back(cells[i].feedback);
	}

	return output;
}

// errorprime must be a vector with length of number of cells
vector<double> MemoryBlock::backward(vector<double> errorPrime, double learningRate) {
	vector<double> eta,
			inputDataPartialSum(connections, 0),
			forgetDataPartialSum(connections, 0);
	double blockSum = 0,
			inputFeedbackPartialSum = 0,
			inputStatePartialSum = 0,
			forgetFeedbackPartialSum = 0,
			forgetStatePartialSum = 0;

	for (unsigned int i = 0; i < cells.size(); i++) {
		blockSum += cells[i].activationOut * errorPrime[i];
		eta.push_back(output * cells[i].activationOutPrime * errorPrime[i]);
		outputFeedbackWeight[i] -= learningRate * blockSum * outputPrime * cells[i].feedback;
		outputStateWeight[i] -= learningRate * blockSum * outputPrime * cells[i].state;
	}

	for (unsigned int i = 0; i < outputDataWeight.size() && i < impulse.size(); i++) {
		outputDataWeight[i] -= learningRate * blockSum * outputPrime * impulse[i];	// invalid read of size 8
	}

	// calculate the updates, and update the cell weights
	for (unsigned int i = 0; i < cells.size(); i++) {
		for (int j = 0; j < connections; j++) {
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
	for (unsigned int i = 0; i < cells.size(); i++) {
		for (int j = 0; j < connections; j++) {
			forgetDataWeight[j] -= learningRate * forgetDataPartialSum[j];	// invalid read of size 8
			inputDataWeight[j] -= learningRate * inputDataPartialSum[j];	// invalid read of size 8
		}
		inputFeedbackWeight[i] -= learningRate * inputFeedbackPartialSum;
		inputStateWeight[i] -= learningRate * inputFeedbackPartialSum;
		forgetFeedbackWeight[i] -= learningRate * forgetFeedbackPartialSum;
		forgetStateWeight[i] -= learningRate * forgetStatePartialSum;
	}

	// truncate error from flowing back, or let error flow
	/*vector<double> weightedError;
	for (unsigned int i = 0; i < cells.size(); i++) {
		for (unsigned int j = 0; j < cells[i].cellDataWeight.size(); j++) {
			if (i == 0)weightedError.push_back(0);
			weightedError[j] += ((partialSum * input * cells[i].activationInPrime) * cells[i].cellDataWeight[j]);
		}
	}*/

	vector<double> temp;	// runs out of memory
	for (unsigned int i = 0; i < cells[0].cellDataWeight.size(); i++) {
		temp.push_back(0.0);
	}

	//return weightedError;
	return temp;
}

