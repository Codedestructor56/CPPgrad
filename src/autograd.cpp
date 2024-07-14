#include <autograd.h>
#include <cmath>
#include <unordered_set>
#include <algorithm>
#include <cstdlib>

// Custom assertion handler
//#define assert(condition) \
    ((condition) ? static_cast<void>(0) : custom_assert(#condition, __FILE__, __LINE__))

/*void custom_assert(const char* condition, const char* file, int line) {
    std::cerr << "Assertion failed: (" << condition << "), file " << file << ", line " << line << std::endl;
    std::abort();  // or std::exit(EXIT_FAILURE);
}

template <typename T>
bool is_int_or_float(T value) {
    return std::is_integral<T>::value || std::is_floating_point<T>::value;
}*/


Data::Data(const double& data):data(data),grad(0.0f), _backward(nullptr), children(){};
Data::Data(const double& data, const vector<Data*>& children):data(data),grad(0.0f), _backward(nullptr), children(children){};


void Data::SetData(const double& data){
  this->data = data;
}

double Data::getData(){
  return data;
}

double Data::getGrad(){
  return grad;
}

void Data::SetGrad(const double& grad){
  this->grad = grad;
}

Data Data::operator+(const double& value) {
    Data* val = new Data(value);
    Data result(this->data + value, vector<Data*>{this, val});
    result._backward = [this, val, &result]() {
        this->grad += 1.0f * result.grad;
        val->grad += 1.0f * result.grad;
    };
    return result;
}

Data Data::operator+(Data& obj) {
    Data result(this->data + obj.data, vector<Data*>{this, &obj});
    result._backward = [this, &obj, &result]() {
        if (this) this->grad += 1.0f * result.grad;
        if (&obj) obj.grad += 1.0f * result.grad;
    };
    return result;
}

Data operator+(const double& value, Data& obj) {
    Data* val = new Data(value);
    Data result(value + obj.data, vector<Data*>{val, &obj});
    result._backward = [&obj, val, &result]() {
        if (val) val->grad += 1.0f * result.grad;
        if (&obj) obj.grad += 1.0f * result.grad;
    };
    return result;
}

Data Data::operator*(const double& value){
    Data* val = new Data(value);
    Data result(this->data * value, vector<Data*>{this, val});
    result._backward = [this, val, &result]() {
        this->grad += val->data * result.grad;
        val->grad += this->data * result.grad;
    };
    return result;

}

Data Data::operator*(Data& obj){
    Data result(this->data * obj.data, vector<Data*>{this, &obj});
    result._backward = [this, &obj, &result]() {
        if (this) this->grad += obj.data * result.grad;
        if (&obj) obj.grad += this->data * result.grad;
    };
    return result;

}

 Data operator*(const double& value, Data& obj){
    Data* val = new Data(value);
    Data result(value * obj.data, vector<Data*>{val, &obj});
    result._backward = [&obj, val, &result]() {
        if (val) val->grad += obj.data * result.grad;
        if (&obj) obj.grad += val->data * result.grad;
    };
    return result;
 }


Data Data::operator^(const double& value){ 
    Data result(pow(this->data, value), vector<Data*>{this});
    result._backward = [this, value, &result]() {
      this->grad += result.grad * value * pow(this->data, value-1.0f);
    };
    return result;

}

Data Data::operator^(Data& obj){
    Data result(pow(this->data, obj.data), vector<Data*>{this});
    result._backward = [this, &obj, &result]() {
        if (this) this->grad += result.grad * obj.data * pow(this->data, obj.data-1.0f);
    };
    return result;
}

Data operator^(const double& value, Data& obj){
    Data* val = new Data(value);
    Data result(pow(value, obj.data), vector<Data*>{val});
    result._backward = [&obj, val, &result]() {
        if (val) val->grad += obj.data * result.grad * pow(val->data, obj.data); 
    };
    return result;
 }

Data Data::operator-(const double& value) {
    return *this + (-value);
}

Data Data::operator-(Data& obj) {
    obj.data *= -1.0f;
    return *this + obj;
}

 Data operator-(const double& value, Data& obj){
    obj.data *= -1.0f;
    return value + obj;
}

Data Data::operator/(const double& value){
    return *this * (pow(value,-1.0f));
}

Data Data::operator/(Data& obj) {
    Data result(this->data / obj.data, vector<Data*>{this, &obj});
    result._backward = [this, &obj, &result]() {
        // Gradient update for this (a in a / b)
        this->grad += (1.0 / obj.data) * result.grad;
        // Gradient update for obj (b in a / b)
        obj.grad -= (this->data / (obj.data * obj.data)) * result.grad;
    };
    return result;
}

Data operator/(const double& value, Data& obj) {
    Data* val = new Data(value);
    Data result(value / obj.data, vector<Data*>{val, &obj});
    result._backward = [&obj, val, &result]() {
        // Gradient update for this (a in a / b)
        val->grad += (1.0 / obj.data) * result.grad;
        // Gradient update for obj (b in value / b)
        obj.grad -= (val->data / (obj.data * obj.data)) * result.grad;
    };
    return result;
}

Data Data::relu() {
    Data result(this->data > 0 ? this->data : 0, std::vector<Data*>{this});
    result._backward = [this, &result]() {
        this->grad += (result.data > 0 ? 1 : 0) * result.grad;
    };
    return result;
}

Data Data::relu(const double& value){
  return Data(value > 0 ? value : 0);
}

Data Data::tanh() {
    double tanh_val = std::tanh(this->data);
    Data result(tanh_val, std::vector<Data*>{this});
    result._backward = [this, tanh_val, &result]() {
        this->grad += (1 - tanh_val * tanh_val) * result.grad;
    };
    return result;
}

Data Data::tanh(const double& value){
  return Data(std::tanh(value));
}

Data Data::sigmoid() {
    double sig_val = 1 / (1 + std::exp(-this->data));
    Data result(sig_val, std::vector<Data*>{this});
    result._backward = [this, sig_val, &result]() {
        this->grad += sig_val * (1 - sig_val) * result.grad;
    };
    return result;
}

Data Data::sigmoid(const double& value){
    double sig_val = 1 / (1 + std::exp(-value));
    Data result(sig_val);
    return result;
}

Data Data::gelu() {
    double gelu_val = 0.5 * this->data * (1 + std::tanh(std::sqrt(2 / M_PI) * (this->data + 0.044715 * std::pow(this->data, 3))));
    Data result(gelu_val, std::vector<Data*>{this});
    result._backward = [this, gelu_val, &result]() {
        double tanh_term = std::tanh(0.035677 * std::pow(this->data, 3) + 0.797885 * this->data);
        double gelu_grad = 0.5 * tanh_term + (0.0535161 * std::pow(this->data, 3) + 0.398942 * this->data) * (1 - std::pow(tanh_term, 2)) + 0.5;
        this->grad += gelu_grad * result.grad;
    };
    return result;
}

Data Data::gelu(const double& value){
    double gelu_val = 0.5 * value * (1 + std::tanh(std::sqrt(2 / M_PI) * (value + 0.044715 * std::pow(value, 3))));
    Data result(gelu_val); 
    return result;
}

Data Data::swish() {
    double sig_val = 1 / (1 + std::exp(-this->data));
    double swish_val = this->data * sig_val;
    Data result(swish_val, std::vector<Data*>{this});
    result._backward = [this, sig_val, &result]() {
        this->grad += (sig_val + this->data * sig_val * (1 - sig_val)) * result.grad;
    };
    return result;
}

Data Data::swish(const double& value){
    double sig_val = 1 / (1 + std::exp(value));
    double swish_val = value * sig_val;
    Data result(swish_val);
    return result;
}
void Data::backwards(){
    std::vector<Data*> topo;
    std::unordered_set<Data*> visited;

    std::function<void(Data*)> build_topo = [&](Data* v) {
        if (visited.find(v) == visited.end()) {
            visited.insert(v);
            for (Data* child : v->getChildren()) {
                build_topo(child);
            }
            topo.push_back(v);
        }
    };

    // Build the topological order starting from this node
    build_topo(this);

    // Set the gradient of the output node
    this->grad = 1.0;

    // Apply the chain rule to get gradients for each node in reverse topological order
    reverse(topo.begin(), topo.end());
    for (Data* v : topo) {
        if(v->_backward){
          v->_backward();
        }
        else{
          v->grad = 0.0f;
        }
    } 
}
