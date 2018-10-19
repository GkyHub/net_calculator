#include "model.hpp"
#include "net.hpp"
#include "typedef.hpp"

Model vgg11();

int main()
{
    Model m = vgg11();
    return 0;
}

Model vgg11()
{
    Model vgg11;
    Net *input = vgg11.add(Input({224, 224}));
    Net *conv1 = vgg11.add(Conv2D("conv1", {input}, {64, 3, 3}, {1, 1}, nl_t::RELU));
    Net *pool1 = vgg11.add(Pool("pool1", conv1, {2, 2}, {2, 2}));
    return vgg11;
}