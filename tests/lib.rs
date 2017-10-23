#[macro_use]
extern crate rulinalg;
extern crate rusty_machine as rm;
extern crate num as libnum;

pub mod learning {
    mod dbscan;
    mod lin_reg;
    mod k_means;
    mod gp;
    mod agglomerative;

    pub mod optim {
    	mod grad_desc;
    }
}

pub mod datasets;