#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include <functional>
#include <vector>
using namespace std;


class Data{
  public:
    Data(const double& data);
    Data(const double& data, const vector<Data*>& children);
    double getData();
    double getGrad();
    void SetGrad(const double & grad);
    void SetData(const double& data);

    Data operator+(const double& value);
    Data operator+(Data& obj);
    Data operator*(const double& value);
    Data operator*(Data& obj);
    Data operator^(const double& value);
    Data operator^(Data& obj);
    Data operator-(const double& value);
    Data operator-(Data& obj);
    Data operator/(const double& value);
    Data operator/(Data& obj);

    friend Data operator+(const double& value, Data& obj);
    friend Data operator*(const double& value, Data& obj);
    friend Data operator^(const double& value, Data& obj);
    friend Data operator-(const double& value, Data& obj);
    friend Data operator/(const double& value, Data& obj);
  
    //activation functions now(we need to define their derivatives individually)
    Data sigmoid();
    static Data sigmoid(const double& value);
    Data relu();
    static Data relu(const double& value);
    Data tanh();
    static Data tanh(const double& value);
    Data swish();
    static Data swish(const double& value);
    Data gelu();
    static Data gelu(const double& value);

    vector<Data*> getChildren() const {
      return children;
    }

    void backwards();
    function<void()> _backward;
  private:
    vector<Data*> children; 
    double data;
    double grad;
    //function<void()> _backward;
};

#endif




