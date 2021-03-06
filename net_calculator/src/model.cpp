#include "model.hpp"
#include "util.hpp"
#include <fstream>

Model::Model()
{
    // empty
}

Model::~Model()
{
    if (!_layers.empty()) {
        for (auto l : _layers) {
            delete l;
        }
    }
}

void Model::EvalStaticSparsity()
{
    for (int32_t i = _layers.size() - 1; i >= 0; i--) {
        _layers[i]->forwardStaticSparsity();
    }

    for (int32_t i = 0; i < _layers.size(); i++) {
        _layers[i]->backwardStaticSparsity();
    }
    return;
}


void Model::Profile(std::string file_name, std::set<Net::type_t> layer_filter)
{
    std::ofstream os(file_name, std::ios::out);
    int n = _layers.size();

    std::vector<double> inference_mac(n);
    std::vector<double> propagation_mac(n);
    std::vector<double> update_mac(n);
    std::vector<double> weight_volume(n);
    std::vector<double> input_volume(n);

    int i = 0;
    for (i = 0; i < n; i++) {
        inference_mac[i] = _layers[i]->getInferenceMacNum();
        weight_volume[i] = _layers[i]->getParamNum();
        input_volume[i] = _layers[i]->getInputSize();
    }

    for (i = n - 1; i >= 0; i--) {
        propagation_mac[i] = _layers[i]->getPropagationMacNum();
        update_mac[i] = _layers[i]->getUpdateMacNum();
    }

    csvPrintLn(os, "", "inference", "propagation", "update", "weight", "input");

    for (i = 0; i < n; i++) {
        csvPrintLn(os, 
            _layers[i]->name, 
            inference_mac[i], 
            propagation_mac[i], 
            update_mac[i], 
            weight_volume[i], 
            input_volume[i]);
    }

	csvPrintLn(os, "Total",
		unit_str(sum(inference_mac)),
		unit_str(sum(propagation_mac)),
		unit_str(sum(update_mac)),
		unit_str(sum(weight_volume)));


    return;
}