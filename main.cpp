/**
 *
 * A program to test a Sawtooth Neural Network
 * Author: Brandon Trabucco
 * Date: 2016/07/27
 *
 */

#include "LSTMNetwork.h"
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>
using namespace std;

long long getMSec() {
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return tp.tv_sec * 1000 + tp.tv_usec / 1000;
}

struct tm *getDate() {
	time_t t = time(NULL);
	struct tm *timeObject = localtime(&t);
	return timeObject;
}

typedef struct {
	int inputSize = 6;
	int inputLength = 6;
	vector<vector<double> > sequence1 = {
			{1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
			{0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
			{0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
			{0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
			{0.0, 0.0, 0.0, 0.0, 1.0, 0.0},
			{0.0, 0.0, 0.0, 0.0, 0.0, 1.0} };
	vector<vector<double> > target1 = {
		{ 1.0 },
		{ 1.0 },
		{ 1.0 },
		{ 1.0 },
		{ 1.0 },
		{ 1.0 } };
	vector<vector<double> > sequence2 = {
			{0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
			{0.0, 0.0, 0.0, 0.0, 1.0, 0.0},
			{0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
			{0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
			{0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
			{1.0, 0.0, 0.0, 0.0, 0.0, 0.0} };
	vector<vector<double> > target2 = {
		{ -1.0 },
		{ -1.0 },
		{ -1.0 },
		{ -1.0 },
		{ -1.0 },
		{ -1.0 } };
} Dataset;

int main(int argc, char *argv[]) {
	cout << "Program initializing" << endl;
	if (argc < 5) {
		cout << argv[0] << " <learning rate> <decay rate> <blocks> <cells> <size ...>" << endl;
		return -1;
	}

	int updatePoints = 100;
	int savePoints = 10;
	int maxEpoch = 1000;
	int blocks = atoi(argv[3]);
	int cells = atoi(argv[4]);
	double errorBound = 0.01;
	double mse1 = 0, mse2 = 0;
	double learningRate = atof(argv[1]), decayRate = atof(argv[2]);
	long long networkStart, networkEnd, sumTime = 0, iterationStart;


	/**
	 *
	 * 	Open file streams to save data samples from Neural Network
	 * 	This data can be plotted via GNUPlot
	 *
	 */
	ostringstream errorDataFileName;
	errorDataFileName << "/u/trabucco/Desktop/Temporal_Convergence_Data_Files/" <<
			(getDate()->tm_year + 1900) << "-" << (getDate()->tm_mon + 1) << "-" << getDate()->tm_mday <<
			"_Single-Core-LSTM-Error_" << learningRate <<
			"-learning_" << decayRate << "-decay.csv";
	ofstream errorData(errorDataFileName.str(), ios::app);
	if (!errorData.is_open()) return -1;


	Dataset dataset;
	LSTMNetwork network = LSTMNetwork((dataset.inputSize), blocks, cells, learningRate, decayRate);


	for (int i = 0; i < (argc - 5); i++) {
		network.addLayer(atoi(argv[5 + i]));
	} network.addLayer(1);


	for (int e = 0; (e < maxEpoch)/* && (!e || (((mse1 + mse2)/2) > errorBound))*/; e++) {
		vector<double> error;

		for (int i = 0; i < dataset.inputLength; i++) {
			if (i == (dataset.inputLength - 1)) error = network.train(dataset.sequence1[i], dataset.target1[0]);
			else network.classify(dataset.sequence1[i]);
		}

		mse1 = 0;
		for (int i = 0; i < error.size(); i++)
			mse1 += error[i] * error[i];
		mse1 /= error.size() * 2;

		for (int i = 0; i < dataset.inputLength; i++) {
			if (i == (dataset.inputLength - 1)) error = network.train(dataset.sequence2[i], dataset.target2[0]);
			else network.classify(dataset.sequence2[i]);
		}

		mse2 = 0;
		for (int i = 0; i < error.size(); i++)
			mse2 += error[i] * error[i];
		mse2 /= error.size() * 2;

		if (((e + 1) % (maxEpoch / updatePoints)) == 0) {
			cout << "Error[" << e << "] = " << ((mse1 + mse2)/2) << endl;
		} errorData << e << ", " << ((mse1 + mse2)/2) << endl;
	}

	errorData.close();

	return 0;
}
