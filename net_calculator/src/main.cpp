#include "model.hpp"
#include "net.hpp"
#include "typedef.hpp"
#include "../example/vgg.hpp"
#include "../example/resnet.hpp"

int main()
{
	ResNet resnet18 = ResNet({ 2, 2, 2, 2 }, { 64, 128, 256, 512 });
	resnet18.Profile("resnet18.csv");
	return 0;
}