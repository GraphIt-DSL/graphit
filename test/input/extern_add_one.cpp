//
// Created by Yunming Zhang on 5/16/19.
//

extern float * vector_a;

void extern_add_one(int v) {
    vector_a[v] += 1;
}