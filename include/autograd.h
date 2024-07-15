#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include <functional>
#include <vector>
using namespace std;

class Data {
public:
    Data(const double& data);
    Data(const double& data, const vector<Data*>& children);
    double getData() const;
    double getGrad() const;
    void setGrad(const double& grad);
    void setData(const double& data);

    Data operator+(const double& value) const;
    Data operator+(const Data& obj) const;
    Data operator*(const double& value) const;
    Data operator*(const Data& obj) const;
    Data operator^(const double& value) const;
    Data operator^(const Data& obj) const;
    Data operator-(const double& value) const;
    Data operator-(const Data& obj) const;
    Data operator/(const double& value) const;
    Data operator/(const Data& obj) const;

    friend Data operator+(const double& value, const Data& obj);
    friend Data operator*(const double& value, const Data& obj);
    friend Data operator^(const double& value, const Data& obj);
    friend Data operator-(const double& value, const Data& obj);
    friend Data operator/(const double& value, const Data& obj);

    // Activation functions
    Data sigmoid() const;
    static Data sigmoid(const double& value);
    Data relu() const;
    static Data relu(const double& value);
    Data tanh() const;
    static Data tanh(const double& value);
    Data swish() const;
    static Data swish(const double& value);
    Data gelu() const;
    static Data gelu(const double& value);

    vector<Data*> getChildren() const { return children; }
    void backward();
    function<void()> _backward;

private:
    vector<Data*> children;
    double data;
    double grad;
};

#endif // AUTOGRAD_H

