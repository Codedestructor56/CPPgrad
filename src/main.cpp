#include <iostream>
#include <autograd.h>
#include <unordered_set>
using namespace std;

void safe_backward(Data& data, const string& name) {
    if (data._backward) {
        cout << "Calling _backward for " << name << endl;
        data._backward();
    } else {
        cout << "_backward is not set for " << name << endl;
    }
}


void traverseGraph(Data* node, unordered_set<Data*>& visited) {
    if (!node || visited.find(node) != visited.end()) {
        return;
    }

    // Mark the current node as visited
    visited.insert(node);

    // Process the current node
    cout << "Node data: " << node->getData() << ", Grad: " << node->getGrad() << endl;

    // Recursively traverse the children
    for (Data* child : node->getChildren()) {
        traverseGraph(child, visited);
    }
}

int main() {
    /*test cases*/
    Data* data = new Data(2.0f);
    Data* data1 = new Data(3.0f);
    cout << data->getData() << endl;
    data->SetData(4.0f);
    cout << data->getData() << endl;
    cout << ((*data) / (*data1)).getData() << endl;

    Data a = Data(2.0f);
    Data c = 2.0f - a;
    Data d = Data(5.0f);
    Data e = d - c;
    Data q = d.tanh();
    Data f = e^3.0f;
    Data r = Data(5.0f);
    Data p = f/r;
    p.backwards(); 

    unordered_set<Data*> visited;
    cout << "Graph Traversal:" << endl;
    traverseGraph(&p, visited);
   
    
    Data ab = Data(3.0f);
    Data cb = Data(5.0f);
    Data nf = Data(11.0f);
    Data po = ab ^ cb;
    Data dpo = po / nf;
    Data mn = dpo.sigmoid();
    cout<< "mn: "<<mn.getData()<<endl;
    cout<<"grad: "<<dpo.getGrad()<<endl;
    mn.backwards();
    unordered_set<Data*> visited1;
    cout << "Graph Traversal:" << endl;
    traverseGraph(&mn, visited);
    Data l = Data(2.0f);
    Data m = l + l;
    m.SetGrad(1.0f);
    safe_backward(m,"m");
    cout << "l: " << l.getGrad() << endl;
    delete data;
    delete data1;

    return 0;
}
