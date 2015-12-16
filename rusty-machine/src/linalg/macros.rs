//! Macros for the linear algebra modules.

/// Should be able to do the following:
///
/// # Specification
///
/// ```
/// let a = mat![1,2,3] // 1 row, 3 cols
/// let b = mat![1;2;3] // 3 rows, 1 col
/// let c = mat![1,2;3,4;5,6] // 3 rows, 2 cols
/// let d = mat![Vector(5), Matrix(5 rows)] // +1 cols
/// let e = mat![Vector(3); Matrix(3 cols)] // +1 rows
/// let f = mat![Matrix(2,3); Matrix(10,3)] // 12 rows, 3 cols
/// let g = mat![Matrix(5,2), Matrix(5,10)] // 5 rows, 12 cols
/// ```
///
/// # Current Support
///
/// This macro currently supports the use cases described
/// by a,b,c in the specification above. 
macro_rules! mat {
	( $($( $x:expr ),*);* ) => {
		{
			let mut vec = Vec::new();
			let mut rows = 0;
			let mut cols = 0;
			
			let mut started_row = false;
			
			$(  
			    let mut inter_cols = 0;
			    
				$(
				    inter_cols += 1;
				    vec.push($x);
				)*
				rows += 1;
				
				if !started_row {
				    cols = inter_cols;
				}
				
				if cols != inter_cols {
				    panic!("Must have equal numbers of elements in each row.");
				}
				
				started_row = true;
			)*
			Matrix { cols : vec.len()/rows, rows: rows, data: vec}
        }
    };
}