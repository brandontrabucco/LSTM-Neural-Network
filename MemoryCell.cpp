/*
 * MemoryCell.cpp
 *
 *  Created on: Jul 18, 2016
 *      Author: trabucco
 */

#include "MemoryCell.h"

long long MemoryCell::n = 0;

MemoryCell::MemoryCell(int connections) :
		cellDataWeight(connections) {
	// TODO Auto-generated constructor stub
	/*activationIn = 0; activationInPrime = 0;
	activationOut = 0; activationOutPrime = 0;
	state = 0; previousState = 0;
	feedback = 0; previousFeedback = 0;
	bias = 0;*/
	nConnections = connections;
	internalError = 0;

	cellFeedbackPartial.push_back(0);
	inputFeedbackPartial.push_back(0);
	inputStatePartial.push_back(0);
	forgetFeedbackPartial.push_back(0);
	forgetStatePartial.push_back(0);

	state.push_back(0);
	feedback.push_back(0);
	previousState.push_back(0);
	previousFeedback.push_back(0);

	bias = 0;

	default_random_engine g(time(0) + (n++));
	normal_distribution<double> d(0, 1);

	cellFeedbackWeight = d(g);

	//cout << "connections: " << connections << endl;
	cellDataPartial.push_back(vector<double>(connections, 0));
	forgetDataPartial.push_back(vector<double>(connections, 0));
	inputDataPartial.push_back(vector<double>(connections, 0));

	//cout << "connections: " << cellDataPartial[0].size() << endl;

	for (int i = 0; i < connections; i++) {
		cellDataWeight[i] = (d(g));
	}
}

MemoryCell::MemoryCell() {
	// TODO Auto-generated destructor stub
}

MemoryCell::~MemoryCell() {
	// TODO Auto-generated destructor stub
}

double MemoryCell::activateIn(double data) {
	double aIn = activationFunction(data);
	activationIn.push_back(aIn);
	activationInPrime.push_back(activationFunctionPrime(data));
	return aIn;
}

double MemoryCell::activateOut(double data) {
	double aOut = activationFunction(data);
	activationOut.push_back(aOut);
	activationOutPrime.push_back(activationFunctionPrime(data));
	return aOut;
}

void MemoryCell::clear() {
	activationIn.clear();
	activationInPrime.clear();
	activationOut.clear();
	activationOutPrime.clear();
	state.clear();
	previousState.clear();
	feedback.clear();
	previousFeedback.clear();

	cellFeedbackPartial.clear();
	inputFeedbackPartial.clear();
	inputStatePartial.clear();
	forgetFeedbackPartial.clear();
	forgetStatePartial.clear();

	cellDataPartial.clear();
	forgetDataPartial.clear();
	inputDataPartial.clear();

	cellDataPartial.push_back(vector<double>(nConnections, 0));
	forgetDataPartial.push_back(vector<double>(nConnections, 0));
	inputDataPartial.push_back(vector<double>(nConnections, 0));

	cellFeedbackPartial.push_back(0);
	inputFeedbackPartial.push_back(0);
	inputStatePartial.push_back(0);
	forgetFeedbackPartial.push_back(0);
	forgetStatePartial.push_back(0);

	state.push_back(0);
	feedback.push_back(0);
	previousState.push_back(0);
	previousFeedback.push_back(0);
}

