use ndarray::*;
use ndarray_linalg::*;

fn adj_matrix_to_Laplacian(adj_matrix: &Array2<u64>) -> Array2<f64> {
    let adj_matrix = adj_matrix.mapv(|x| x as f64);
    let degree: Array1<f64> = adj_matrix.sum_axis(Axis(1));
    let d_1_2 = degree.mapv(|x| 1.0/(f64::sqrt(x)));
    let D_1_2 = Array::eye( d_1_2.len()) * &d_1_2;
    Array::eye(d_1_2.len()) - D_1_2.dot(&adj_matrix.dot(&D_1_2))
}

fn matrix_to_feature(m: &Array2<f64>, features: u64) -> Array1<f64>{
    let (e, _) = m.clone().eigh(UPLO::Lower).unwrap();
    e
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_adj_matrix_to_Laplacian(){
        let adj_matrix:Array2<u64> = array![[0,1,1,1,1],
                                            [1,0,0,0,0],
                                            [1,0,0,0,0],
                                            [1,0,0,0,0],
                                            [1,0,0,0,0]];


        let laplacian_matrix:Array2<f64> = array![[ 1.0,-0.5,-0.5,-0.5,-0.5],
                                                  [-0.5, 1.0, 0.0, 0.0, 0.0],
                                                  [-0.5, 0.0, 1.0, 0.0, 0.0],
                                                  [-0.5, 0.0, 0.0, 1.0, 0.0],
                                                  [-0.5, 0.0, 0.0, 0.0, 1.0]];

        assert_eq!(adj_matrix_to_Laplacian(&adj_matrix),  laplacian_matrix);
    }
}
