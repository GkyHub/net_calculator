#include "model.hpp"
#include "net.hpp"
#include "typedef.hpp"
#include "vgg11.hpp"

Model *vgg11();

int main()
{
    Model *m = vgg11();
	m->Profile("test.csv");
	delete m;
    return 0;
}