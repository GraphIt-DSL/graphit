//
// Created by Yunming Zhang on 5/16/19.
//

extern float * vector_a;

void extern_src_add_one(int src, int dst) {
    vector_a[src] += 1;
}