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
Data::Data(const double& data) : data(data), grad(0.0), _backward(nullptr), children() {}
Data::Data(const double& data, const vector<Data*>& children) : data(data), grad(0.0), _backward(nullptr), children(children) {}

void Data::setData(const double& data) {
    this->data = data;
}

double Data::getData() const{
    return data;
}

double Data::getGrad() const{
    return grad;
}

void Data::setGrad(const double& grad) {
    this->grad = grad;
}

Data Data::operator+(const double& value) const {
    Data* val = new Data(value);
    Data result(this->data + value, vector<Data*>{const_cast<Data*>(this), val});
    result._backward = [this, val, &result]() {
        const_cast<Data*>(this)->grad += 1.0 * result.grad;
        val->grad += 1.0 * result.grad;
    };
    return result;
}

Data Data::operator+(const Data& obj) const {
    Data result(this->data + obj.data, vector<Data*>{const_cast<Data*>(this), const_cast<Data*>(&obj)});
    result._backward = [this, &obj, &result]() {
        const_cast<Data*>(this)->grad += 1.0 * result.grad;
        const_cast<Data*>(&obj)->grad += 1.0 * result.grad;
    };
    return result;
}

Data operator+(const double& value, const Data& obj) {
    Data* val = new Data(value);
    Data result(value + obj.data, vector<Data*>{val, const_cast<Data*>(&obj)});
    result._backward = [&obj, val, &result]() {
        val->grad += 1.0 * result.grad;
        const_cast<Data*>(&obj)->grad += 1.0 * result.grad;
    };
    return result;
}

//this tiny bit of code has been a pain in my ass, just do not change it
Data& Data::operator+=(const Data& other) {
    Data* original = new Data(*this);  // Store the current state of *this
    Data* other_copy = new Data(other);  // Create a copy of other

    this->data += other.data;  // Update the data value

    this->children = {original, other_copy};  // Update the children to include original state and other

    this->_backward = [this, original, other_copy]() {
        original->grad += 1.0 * this->grad;
        other_copy->grad += 1.0 * this->grad;
    };

    return *this;
}

Data Data::operator*(const double& value) const {
    Data* val = new Data(value);
    Data result(this->data * value, vector<Data*>{const_cast<Data*>(this), val});
    result._backward = [this, val, &result]() {
        const_cast<Data*>(this)->grad += val->data * result.grad;
        val->grad += this->data * result.grad;
    };
    return result;
}

Data Data::operator*(const Data& obj) const {
    Data result(this->data * obj.data, vector<Data*>{const_cast<Data*>(this), const_cast<Data*>(&obj)});
    result._backward = [this, &obj, &result]() {
        const_cast<Data*>(this)->grad += obj.data * result.grad;
        const_cast<Data*>(&obj)->grad += this->data * result.grad;
    };
    return result;
}

Data operator*(const double& value, const Data& obj) {
    Data* val = new Data(value);
    Data result(value * obj.data, vector<Data*>{val, const_cast<Data*>(&obj)});
    result._backward = [&obj, val, &result]() {
        val->grad += obj.data * result.grad;
        const_cast<Data*>(&obj)->grad += val->data * result.grad;
    };
    return result;
}

Data& Data::operator*=(const Data& other) {
    Data* original = new Data(*this);
    Data* other_copy = new Data(other);  

    this->data += other.data; 

    this->children = {original, other_copy}; 

    this->_backward = [this, original, other_copy]() {
        original->grad += other_copy->grad * this->grad;
        other_copy->grad += original->grad * this->grad;
    };

    return *this;
}

Data Data::operator^(const double& value) const {
    Data result(std::pow(this->data, value), vector<Data*>{const_cast<Data*>(this)});
    result._backward = [this, value, &result]() {
        const_cast<Data*>(this)->grad += result.grad * value * std::pow(this->data, value - 1.0);
    };
    return result;
}

Data Data::operator^(const Data& obj) const {
    Data result(std::pow(this->data, obj.data), vector<Data*>{const_cast<Data*>(this), const_cast<Data*>(&obj)});
    result._backward = [this, &obj, &result]() {
        const_cast<Data*>(this)->grad += result.grad * obj.data * std::pow(this->data, obj.data - 1.0);
        const_cast<Data*>(&obj)->grad += result.grad * std::log(this->data) * std::pow(this->data, obj.data);
    };
    return result;
}

Data operator^(const double& value, const Data& obj) {
    Data* val = new Data(value);
    Data result(std::pow(value, obj.data), vector<Data*>{val, const_cast<Data*>(&obj)});
    result._backward = [&obj, val, &result]() {
        val->grad += result.grad * obj.data * std::pow(val->data, obj.data - 1.0);
        const_cast<Data*>(&obj)->grad += result.grad * std::log(val->data) * std::pow(val->data, obj.data);
    };
    return result;
}

Data Data::operator-(const double& value) const {
    Data* val = new Data(value);
    Data result(this->data - value, vector<Data*>{const_cast<Data*>(this), val});
    result._backward = [this, val, &result]() {
        const_cast<Data*>(this)->grad += 1.0 * result.grad;
        val->grad += -1.0 * result.grad;
    };
    return result;
}

Data Data::operator-(const Data& obj) const {
    Data result(this->data - obj.data, vector<Data*>{const_cast<Data*>(this), const_cast<Data*>(&obj)});
    result._backward = [this, &obj, &result]() {
        const_cast<Data*>(this)->grad += 1.0 * result.grad;
        const_cast<Data*>(&obj)->grad += -1.0 * result.grad;
    };
    return result;
}

Data operator-(const double& value, const Data& obj) {
    Data* val = new Data(value);
    Data result(value - obj.data, vector<Data*>{val, const_cast<Data*>(&obj)});
    result._backward = [&obj, val, &result]() {
        val->grad += 1.0 * result.grad;
        const_cast<Data*>(&obj)->grad += -1.0 * result.grad;
    };
    return result;
}

Data Data::operator/(const double& value) const {
    Data* val = new Data(value);
    Data result(this->data / value, vector<Data*>{const_cast<Data*>(this), val});
    result._backward = [this, val, &result]() {
        const_cast<Data*>(this)->grad += (1.0 / val->data) * result.grad;
        val->grad += -(this->data / std::pow(val->data, 2.0)) * result.grad;
    };
    return result;
}

Data Data::operator/(const Data& obj) const {
    Data result(this->data / obj.data, vector<Data*>{const_cast<Data*>(this), const_cast<Data*>(&obj)});
    result._backward = [this, &obj, &result]() {
        const_cast<Data*>(this)->grad += (1.0 / obj.data) * result.grad;
        const_cast<Data*>(&obj)->grad += -(this->data / std::pow(obj.data, 2.0)) * result.grad;
    };
    return result;
}

Data operator/(const double& value, const Data& obj) {
    Data* val = new Data(value);
    Data result(value / obj.data, vector<Data*>{val, const_cast<Data*>(&obj)});
    result._backward = [&obj, val, &result]() {
        val->grad += (1.0 / obj.data) * result.grad;
        const_cast<Data*>(&obj)->grad += -(val->data / std::pow(obj.data, 2.0)) * result.grad;
    };
    return result;
}

// Activation functions
Data Data::sigmoid() const {
    double result = 1.0 / (1.0 + std::exp(-this->data));
    Data res(result, vector<Data*>{const_cast<Data*>(this)});
    res._backward = [this, result, &res]() {
        const_cast<Data*>(this)->grad += result * (1 - result) * res.grad;
    };
    return res;
}

Data Data::relu() const {
    double result = std::max(0.0, this->data);
    Data res(result, vector<Data*>{const_cast<Data*>(this)});
    res._backward = [this, result, &res]() {
        const_cast<Data*>(this)->grad += (result > 0 ? 1 : 0) * res.grad;
    };
    return res;
}

Data Data::tanh() const {
    double result = std::tanh(this->data);
    Data res(result, vector<Data*>{const_cast<Data*>(this)});
    res._backward = [this, result, &res]() {
        const_cast<Data*>(this)->grad += (1 - result * result) * res.grad;
    };
    return res;
}

Data Data::swish() const {
    double sigmoid_result = 1.0 / (1.0 + std::exp(-this->data));
    double result = this->data * sigmoid_result;
    Data res(result, vector<Data*>{const_cast<Data*>(this)});
    res._backward = [this, sigmoid_result, result, &res]() {
        const_cast<Data*>(this)->grad += (sigmoid_result + result * (1 - sigmoid_result)) * res.grad;
    };
    return res;
}

Data Data::gelu() const {
    double result = 0.5 * this->data * (1 + std::tanh(std::sqrt(2 / M_PI) * (this->data + 0.044715 * std::pow(this->data, 3))));
    Data res(result, vector<Data*>{const_cast<Data*>(this)});
    res._backward = [this, result, &res]() {
        const double x = this->data;
        const double tanh_out = std::tanh(0.0356774 * std::pow(x, 3) + 0.797885 * x);
        const_cast<Data*>(this)->grad += (0.5 * (1 + tanh_out) + 0.0356774 * std::pow(x, 3) * (1 - std::pow(tanh_out, 2))) * res.grad;
    };
    return res;
}


Data Data::sigmoid(const double& value) {
    double result = 1.0 / (1.0 + std::exp(-value));
    Data res(result);
    return res;
}

Data Data::relu(const double& value) {
    double result = std::max(0.0, value);
    Data res(result);
    return res;
}

Data Data::tanh(const double& value) {
    double result = std::tanh(value);
    Data res(result);
    return res;
}

Data Data::swish(const double& value) {
    double result = value * sigmoid(value).data;
    Data res(result);
    return res;
}

Data Data::gelu(const double& value) {
    double result = 0.5 * value * (1 + std::tanh(std::sqrt(2 / M_PI) * (value + 0.044715 * std::pow(value, 3))));
    Data res(result);
    return res;
}

void Data::backward() {
    unordered_set<Data*> visited;
    vector<Data*> topo;

    function<void(Data*)> buildTopo = [&](Data* v) {
        if (visited.find(v) == visited.end()) {
            visited.insert(v);
            for (Data* child : v->children) {
                buildTopo(child);
            }
            topo.push_back(v);
        }
    };

    buildTopo(this);
    reverse(topo.begin(), topo.end());

    this->grad = 1.0;
    for (Data* v : topo) {
        if (v->_backward) {
            v->_backward();
        }
    }
}

