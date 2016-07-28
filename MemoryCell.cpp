/*
 * MemoryCell.cpp
 *
 *  Created on: Jul 18, 2016
 *      Author: trabucco
 */

#include "MemoryCell.h"

long long MemoryCell::n = 0;

MemoryCell::MemoryCell(int connections) {
	// TODO Auto-generated constructor stub
	activationIn = 0; activationInPrime = 0;
	activationOut = 0; activationOutPrime = 0;
	state = 0; previousState = 0;
	feedback = 0; previousFeedback = 0;
	bias = 0;

	default_random_engine g(time(0) + (n++));
	normal_distribution<double> d(0, 1);

	cellFeedbackWeight = d(g);
	cellFeedbackPartial = 0;
	inputFeedbackPartial = 0;
	inputStatePartial = 0;
	forgetFeedbackPartial = 0;
	forgetStatePartial = 0;

	for (int i = 0; i < connections; i++) {
		cellDataWeight.push_back(d(g));
		cellDataPartial.push_back(0);
		forgetDataPartial.push_back(0);	// invalid memory
		inputDataPartial.push_back(0);	// invalid memory
	}
}

MemoryCell::~MemoryCell() {
	// TODO Auto-generated destructor stub
}

double MemoryCell::activateIn(double data) {
	activationIn = activationFunction(data);
	activationInPrime = activationFunctionPrime(data);
	return activationIn;
}

double MemoryCell::activateOut(double data) {
	activationOut = activationFunction(data);
	activationOutPrime = activationFunctionPrime(data);
	return activationOut;
}

