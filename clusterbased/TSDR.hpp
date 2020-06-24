#define MAX_DETECTIONS 50
#define MAX_LABELS 20
#define E_CONST 2.71828

#define ALPHA 2.0
#define BETA 1.0

namespace TSDR
{
    class Cluster {
    public:
        int id;
        int x, y, w, h;
        int n; // number of keypoints in cluster
        
        int kp_xmin, kp_ymin;
        int kp_xmax, kp_ymax;
        
        std::list<cv::KeyPoint> associated;
        
        Cluster () {}

        void add_keypoint(cv::KeyPoint kp) {
            this->associated.push_back(kp);
            this->n++;
            int x = int(kp.pt.x);
            int y = int(kp.pt.y);
            if(x<this->kp_xmin) this->kp_xmin=x;
            else if(x>this->kp_xmax) this->kp_xmax=x;
            if(y<this->kp_ymin) this->kp_ymin=y;
            else if(y>this->kp_ymax) this->kp_ymax=y;
        }
        
        Cluster(int id, int x, int y, int w, int h) {
            this->id = id;
            this->x = x;
            this->y = y;
            this->w = w;
            this->h = h;
            
            kp_xmin=INT_MAX;
            kp_xmax=0;
            kp_ymin=INT_MAX;
            kp_ymax=0;
        }
        
    };
    
    class ROI {
    public:
        int x, y; // x, y coords of top left corner
        int xs, ys; // width, height respectively
        int priority;
        Cluster *cluster;
        int tag = -1;
        
        ROI (int x, int y, int xs, int ys) {
            this->x = x;
            this->y = y;
            this->xs = xs;
            this->ys = ys;
        }
        
        ROI (int x, int y, int xs, int ys, int priority) {
            this->x = x;
            this->y = y;
            this->xs = xs;
            this->ys = ys;
            this->priority = priority;
        }
        
        ROI (int x, int y, int xs, int ys, int priority, Cluster *cluster) {
            this->x = x;
            this->y = y;
            this->xs = xs;
            this->ys = ys;
            this->priority = priority;
            this->cluster = cluster;
        }
    };
    bool compareROIs (ROI a, ROI b) { return a.priority>b.priority; }
    
    class Detection {
    public:
        int object_class;
        double confidence;
        int x,y; // x,y coords of top left corner
        int xs,ys; // width, height respectively
        Cluster* associated_cluster;
        
        Detection (int object_class, double confidence, int x, int y, int xs, int ys, Cluster *cluster) {
            this->object_class = object_class;
            this->confidence = confidence;
            this->x = x;
            this->y = y;
            this->xs = xs;
            this->ys = ys;
            this->associated_cluster = cluster;
        }
        
        Detection (int object_class, double confidence, int x, int y, int xs, int ys) {
            this->object_class = object_class;
            this->confidence = confidence;
            this->x = x;
            this->y = y;
            this->xs = xs;
            this->ys = ys;
        }
    };
    
    class Detection_Profile {
    public:
        // current location
        int x,y;
        int xs,ys;
        
        double reliability;
        int predicted_class;
        
        double data[MAX_LABELS][MAX_DETECTIONS] = {{0.0}};
        
        double reliability_array[MAX_LABELS] = {0.0};
        double softmaxReliability_array[MAX_LABELS] = {0.0};
        
        std::list<Detection*> detections;
        int N_d = 0; // number of times detected
        int ticks_left; // time until it will need to be detected again
        
        void setCounter() {
            ticks_left = int(pow(2,N_d));
        }
        void tick() {
            ticks_left--;
        }
        
        void predict() {
            int i = 0;
            int max_pos = -1;
            double max_val = -1.0;
            
            for(i=0; i< MAX_LABELS; i++) {
                if(softmaxReliability_array[i] > max_val ) {
                    max_pos = i;
                    max_val = softmaxReliability_array[i];
                }
            }
            
            this->reliability = max_val;
            this->predicted_class = max_pos;
        }
        
        void setReliabilityArray() {
            int i;
            for(i=0; i<MAX_LABELS; i++) {
                int j;
                int n = 0;
                double R = 1.0;
                for(j=0; j< MAX_DETECTIONS; j++) {
                    if(j!=0) {
                        n++;
                        R = R*(1.0-data[i][j]);
                    }
                }
                R = 1.0-R;
                reliability_array[i] =(1.0-pow(ALPHA, -BETA*double(n)))*R;
            }
            
            return;
        }
        
        void setSoftmaxArray() {
            int i;
            
            double denominator = 0.0;
            for(i=0; i< MAX_LABELS; i++) {
                if(reliability_array[i]!=0)
                    denominator += pow(E_CONST, reliability_array[i]);
            }
            
            for(i=0; i< MAX_LABELS; i++) {
                if(reliability_array[i]!=0)
                    softmaxReliability_array[i] = reliability_array[i]*pow(E_CONST, reliability_array[i])/denominator;
            }
            return;
        }
        
        
        void add_detection(Detection* _detection) {
            if(N_d>=MAX_DETECTIONS) return; // laziest thing I have done, but it should work
            this->x = _detection->x;
            this->y = _detection->y;
            this->xs = _detection->xs;
            this->ys = _detection->ys;
            
            this->detections.push_back(_detection);
            
            int object_class = _detection->object_class;
            double confidence = _detection->confidence;
            
            printf("class=%d, conf=%f\n", object_class, confidence);
            data[object_class][N_d] = confidence;
            
            N_d++;
            setCounter();
            setReliabilityArray();
            setSoftmaxArray();
            predict();
        }
        
        Detection_Profile(Detection* _detection) {
            this->x = _detection->x;
            this->y = _detection->y;
            this->xs = _detection->xs;
            this->ys = _detection->ys;
            this->detections.push_back(_detection);
            
            int object_class = _detection->object_class;
            double confidence = _detection->confidence;
            
            
            printf("class=%d, conf=%f\n", object_class, confidence);
            data[object_class][N_d] = confidence;
            
            N_d++;
            setCounter();
            setReliabilityArray();
            setSoftmaxArray();
            predict();
        }
        
        void printTable() {
            int i,j;
            /*
            for(j=0; j< MAX_DETECTIONS; j++) {
                for(i=0; i<MAX_LABELS; i++) {
                    printf("%f, ", data[i][j]);
                }
                
                printf("\n");            }

            return;
            */
            for(i=0; i<MAX_LABELS; i++) {
                //printf("%f, ",reliability_array[i]);
                printf("%f, ", softmaxReliability_array[i]);
            }
            printf("\n");
        }
    };
}
