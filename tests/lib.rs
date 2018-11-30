#[macro_use]
extern crate rulinalg;
extern crate num as libnum;
extern crate rusty_machine as rm;

pub mod learning {
    mod dbscan;
    mod gp;
    mod k_means;
    mod knn;
    mod lin_reg;
    mod pca;

    pub mod optim {
        mod grad_desc;
    }
}

pub mod datasets;
