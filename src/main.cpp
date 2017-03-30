#include "jpg.h"
#include "mnist.h"
#include "limits.h"


float distance(int* v1, int* v2){
	float d=0;
	for (int i=0; i<784; i++){
	d+= (v1[i]-v2[i])*(v1[i]-v2[i]);
	}

	return d;
}
float linear_classifier(float* w, float* x){
float d=0;
for(int i=0; i<784;i++){
	d+=w[i]*x[i];
}
if(d>=0) return 1;
else return 0;
}

float distance( int* v1, int*v2);
int main(){

    float** data = read_mnist("train-images.idx3-ubyte"); // readmnist definie dns mnist.h//
    float* labels = read_labels("train-labels.idx1-ubyte"); 
    float** test_images= read_mnist("t10k-images.idx3-ubyte");
    float* test_labels= read_labels("t10k-images.idx3-ubyte");
    float* w= new float[784];
	for(int i=0;i<784; i++) w[i]=(float)rand()*2/INT_MAX-1;
	float E=0; 
    for(int i=0; i<10000; i++){
	printf("%u\n",i);
	
       float inference = linear_classifier(w,test_images[i]);
       save_jpg(test_images[i], 28, 28, "%u/%u.jpg", inference, i);
		if(inference!=test_labels[i]) E++;
		printf("Erreur=%0.2f%%\n", (E*100)/i);
    
}
    return 0;
}

