/**
 *
 * A program to test a LSTM Neural Network
 * Author: Brandon Trabucco
 * Date: 2016/07/27
 *
 */

#include "LSTMNetwork.h"
#include "DatasetAdapter.h"
#include "OutputTarget.h"
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
	double mse = 0;
	double learningRate = atof(argv[1]), decayRate = atof(argv[2]);
	long long networkStart, networkEnd, sumTime = 0, iterationStart;

	const int _day = getDate()->tm_mday;


	/**
	 *
	 * 	Open file streams to save data samples from Neural Network
	 * 	This data can be plotted via GNUPlot
	 *
	 */
	ostringstream errorDataFileName;
	errorDataFileName << "/u/trabucco/Desktop/Temporal_Convergence_Data_Files/" <<
			(getDate()->tm_year + 1900) << "-" << (getDate()->tm_mon + 1) << "-" << _day <<
			"_Single-Core-LSTM-Error_" << learningRate <<
			"-learning_" << decayRate << "-decay.csv";
	ofstream errorData(errorDataFileName.str(), ios::app);
	if (!errorData.is_open()) return -1;


	ostringstream accuracyDataFileName;
	accuracyDataFileName << "/u/trabucco/Desktop/Temporal_Convergence_Data_Files/" <<
			(getDate()->tm_year + 1900) << "-" << (getDate()->tm_mon + 1) << "-" << _day <<
			"_Single-Core-LSTM-Accuracy_" << learningRate <<
			"-learning_" << decayRate << "-decay.csv";
	ofstream accuracyData(accuracyDataFileName.str(), ios::app);
	if (!accuracyData.is_open()) return -1;


	networkStart = getMSec();
	DatasetAdapter dataset = DatasetAdapter();
	networkEnd = getMSec();
	cout << "KTH Dataset loaded in " << (networkEnd - networkStart) << "msecs" << endl;


	LSTMNetwork network = LSTMNetwork(dataset.getFrameSize(), blocks, cells, learningRate, decayRate);
	cout << "Network initialized" << endl;


	for (int i = 0; i < (argc - 5); i++) {
		network.addLayer(atoi(argv[5 + i]));
	} network.addLayer(6);


	for (int e = 0; (e < maxEpoch)/* && (!e || (((mse1 + mse2)/2) > errorBound))*/; e++) {
		vector<double> error;

		networkStart = getMSec();
		while (dataset.nextTrainingVideo()) {
			while (dataset.nextTrainingFrame()) {
				DatasetExample data = dataset.getTrainingFrame();
				if (dataset.isLastTrainingFrame()) {
					error = network.train(data.frame, OutputTarget::getOutputFromTarget(data.label));
				} else network.classify(data.frame);
			}
		}

		int c = 0;
		while (dataset.nextTestVideo()) {
			vector<double> output;
			while (dataset.nextTestFrame()) {
				DatasetExample data = dataset.getTestFrame();
				if (dataset.isLastTrainingFrame()) {
					output = network.classify(data.frame);
					if (OutputTarget::getTargetFromOutput(output) == data.label) c++;
				} else network.classify(data.frame);
			}
		} networkEnd = getMSec();

		mse = 0;
		for (int i = 0; i < error.size(); i++)
			mse += error[i] * error[i];
		mse /= error.size() * 2;

		if (((e + 1) % (maxEpoch / updatePoints)) == 0) {
			cout << "Epoch " << e << " completed in " << (networkEnd - networkStart) << "msecs" << endl;
			cout << "Error[" << e << "] = " << mse << endl;
			cout << "Accuracy[" << e << "] = " << (100.0 * (float)c / (float)dataset.getTestSize()) << endl;
		} errorData << e << ", " << mse << endl;
		accuracyData << e << ", " << (100.0 * (float)c / (float)dataset.getTestSize()) << endl;

		dataset.reset();
	}

	errorData.close();

	return 0;
}
