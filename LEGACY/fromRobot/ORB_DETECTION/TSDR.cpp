namespace TSDR
{
    class ROI {
    public:
        int x, y; // x, y coords of top left corner
        int xs, ys; // width, height respectively
        int priority;
        
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
    };
    
    class Detection {
    public:
        int object_class;
        double confidence;
        int x,y; // x,y coords of top left corner
        int xs,ys; // width, height respectively
        
        Detection (int object_class, double confidence, int x, int y, int xs, int ys) {
            this->object_class = object_class;
            this->confidence = confidence;
            this->x = x;
            this->y = y;
            this->xs = xs;
            this->ys = ys;
        }
        
        void setPosition(int x, int y) {
            this->x = x;
            this->y = y;
        }
        
        void update_horizontals(int x, int xs) {
            this->x = x;
            this->xs = xs;
        }
        
        void update_verticals(int y, int ys) {
            this->y = y;
            this->ys = ys;
        }
        
        void setDimensions(int xs, int ys) {
            this->xs = xs;
            this->ys = ys;
        }
        
        void updateConfidence (double confidence) {
            this->confidence = confidence;
        }
    };
}
