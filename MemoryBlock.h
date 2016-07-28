/*
 * MemoryBlock.h
 *
 *  Created on: Jul 18, 2016
 *      Author: trabucco
 */

#ifndef MEMORYBLOCK_H_
#define MEMORYBLOCK_H_

#include "BaseNode.h"
#include "MemoryCell.h"
#include <vector>
#include <math.h>
#include <time.h>
#include <iostream>
#include <random>
using namespace std;

class MemoryBlock : public BaseNode {
private:
	static long long n;
public:
	int connections;
	vector<MemoryCell> cells;
	vector<double> inputDataWeight,
		forgetDataWeight, outputDataWeight,
		bias, impulse,
		inputFeedbackWeight, inputStateWeight,
		forgetFeedbackWeight, forgetStateWeight,
		outputFeedbackWeight, outputStateWeight;
	double input, inputPrime,
		forget, forgetPrime,
		output, outputPrime;
	double inputGate(double data);
	double forgetGate(double data);
	double outputGate(double data);
	MemoryBlock(int nCells, int connections);
	virtual ~MemoryBlock();
	vector<double> forward(vector<double> input);
	vector<double> backward(vector<double> errorPrime, double learningRate);
};

#endif /* MEMORYBLOCK_H_ */
