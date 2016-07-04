//! Macros for the linear algebra modules.

macro_rules! count {
    () => (0usize);
    ( $x:tt $($xs:tt)* ) => (1usize + count!($($xs)*));
}

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
    ( $( $x:expr ),* ) => { {
        let vec = vec![$($x),*];
        Matrix { cols : vec.len(), rows: 1, data: vec }
    } };
    ( $( $x0:expr ),* ; $($( $x:expr ),*);* ) => { {
        let mut _assert_width0 = [(); count!($($x0)*)];
        let mut vec = Vec::new();
        let rows = 1usize;
        let cols = count!($($x0)*);

        $( vec.push($x0); )*

        $(
            let rows = rows + 1usize;
            let _assert_width = [(); count!($($x)*)];
            _assert_width0 = _assert_width;
            $( vec.push($x); )*
        )*

        Matrix { cols : cols, rows: rows, data: vec }
    } }
}
