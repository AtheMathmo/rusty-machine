extern crate rusty_machine as rm;
extern crate num as libnum;

pub mod linalg {
    mod mat;
}

pub mod learning {
    mod lin_reg;
    mod k_means;
    mod gp;

    pub mod optim {
    	mod grad_desc;
    }
}