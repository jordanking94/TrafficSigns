#define DEFAULT_LIFETIME 2

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
        int ref;
        
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
        
        ROI (int x, int y, int xs, int ys, int priority, int ref) {
            this->x = x;
            this->y = y;
            this->xs = xs;
            this->ys = ys;
            this->priority = priority;
            this->ref = ref;
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
        int x,y;
        int xs,ys;
        double reliability;
        std::list<Detection *> detections;
        int num_detections;
        
        int life_counter;
        
        Detection_Profile() {}
        
        void update_profile() {
            double tmp = pow(2.0, double(num_detections) );
            life_counter = int(tmp);
            
        }
        
        void tick_tock() {
            life_counter--;
        }
        
        void add_detection(Detection* d) {
            this->detections.push_back(d);
            num_detections++;
            update_profile();
        }
        
    };
}
