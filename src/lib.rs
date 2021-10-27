use ndarray::*;
use ndarray_linalg::*;
use std::cmp::min;
use std;
use std::collections::BTreeMap;

use linfa;
use linfa::prelude::*;
use linfa_trees;
use serde_json;

fn adj_matrix_to_Laplacian(adj_matrix: &Array2<u64>) -> Array2<f64> {
    let adj_matrix = adj_matrix.mapv(|x| x as f64);
    let degree: Array1<f64> = adj_matrix.sum_axis(Axis(1));
    let d_1_2 = degree.mapv(|x| 1.0/(f64::sqrt(x)));
    let D_1_2 = Array::eye( d_1_2.len()) * &d_1_2;
    Array::eye(d_1_2.len()) - D_1_2.dot(&adj_matrix.dot(&D_1_2))
}

fn matrix_to_feature(m: &Array2<f64>, features: usize) -> Array1<f64>{
    let e = m.clone().eigvalsh(UPLO::Lower).unwrap();
    let mut ret: Array1<f64> = Array::zeros(features);
    let e: Array1<f64> = e.iter().filter(|&x| *x > 0.0).map(|&x| x).collect();
    let min_element = min(features, e.len());
    ret.slice_mut(s!(0..min_element)).assign(&e.slice(s!(0..min_element)));
    ret 
}

fn adj_matrices_to_features(adj_matrices: &Vec<Array2<u64>>, features: usize) -> Array2<f64>{
    let mut ret:Array2<f64> = Array2::zeros((adj_matrices.len(),features));
    for (idx, val) in adj_matrices.iter().map(|x| matrix_to_feature(&adj_matrix_to_Laplacian(x), features)).enumerate(){
        ret.slice_mut(s!(idx, 0..features)).assign(&val);
    }
    ret
}

fn build_model(split_quality: linfa_trees::SplitQuality, 
             max_depth: Option<usize>,
             min_weight_split: Option<f32>,
             min_weight_leaf: Option<f32>
             ) -> linfa_trees::DecisionTreeParams<f64,usize> {
    let mut model = linfa_trees::DecisionTree::params()
        .split_quality(split_quality)
        .max_depth(max_depth);

    if let Some(i) = min_weight_split {
        model = model.min_weight_split(i);
        }

    if let Some(i) = min_weight_leaf {
        model = model.min_weight_leaf(i);
    }


    model
}

fn load_dataset(adj_list_path: &std::path::PathBuf, 
                node_graph_list_path: &std::path::PathBuf,
                graph_class_path: &Option<std::path::PathBuf>
                ) -> (Vec<Array2<u64>>, Option<Array1<u64>>){
    if !adj_list_path.exists() {
        eprintln!("Path: '{}'  does not exists", adj_list_path.display());
        std::process::exit(1);
    }

    if !node_graph_list_path.exists(){
        eprintln!("Path: '{}'  does not exists", node_graph_list_path.display());
        std::process::exit(1);
    }

    if let Some(path) = graph_class_path {
        if !path.exists(){
            eprintln!("Path: '{}'  does not exists", path.display());
            std::process::exit(1);
        }
    }

    let node_graph_list_file = std::io::BufReader::new(std::fs::File::open(node_graph_list_path).unwrap());
    let node_graph_list: Vec<u64> = csv::ReaderBuilder::new().has_headers(false).trim(csv::Trim::All).from_reader(node_graph_list_file).records().map(|x| x.unwrap()[0].parse::<u64>().unwrap()).collect();

    let mut graphs_size: BTreeMap<u64, usize> = BTreeMap::new();
    for x in node_graph_list.iter(){
        *graphs_size.entry(*x).or_default() += 1;
    }
    let mut dataset: BTreeMap<u64, Array2<u64>> = graphs_size.iter().map(|(k,v)| (*k, Array2::zeros((*v,*v)))).collect();

    let adj_list_file = std::io::BufReader::new(std::fs::File::open(adj_list_path).unwrap());
    for edge in csv::ReaderBuilder::new().has_headers(false).trim(csv::Trim::All).from_reader(adj_list_file).records() {
        let edge = edge.unwrap();
        let n1: usize = edge[0].parse().unwrap();
        let n2: usize = edge[1].parse().unwrap();

        let graph_id:u64 = node_graph_list[n1 - 1];

        let n0 = node_graph_list.iter().position(|&x| x == graph_id).unwrap() + 1;
        dataset.get_mut(&graph_id).unwrap()[[n1 - n0,n2 - n0]] = 1;

    }


    if let Some(path) = graph_class_path {
        let graph_class_file = std::io::BufReader::new(std::fs::File::open(path).unwrap());
        let graph_class: Array1<u64> = csv::ReaderBuilder::new().has_headers(false).trim(csv::Trim::All).from_reader(graph_class_file).records().map(|x| x.unwrap()[0].parse::<u64>().unwrap()).collect();
        return (dataset.iter().map(|(_,v)| v.clone()).collect(), Some(graph_class));
    }
    

    (dataset.iter().map(|(_,v)| v.clone()).collect(), None)

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

    #[test]
    fn test_matrix_to_feature_equal(){
        let laplacian_matrix:Array2<f64> = array![[ 1.0,-0.5,-0.5,-0.5,-0.5],
                                                  [-0.5, 1.0, 0.0, 0.0, 0.0],
                                                  [-0.5, 0.0, 1.0, 0.0, 0.0],
                                                  [-0.5, 0.0, 0.0, 1.0, 0.0],
                                                  [-0.5, 0.0, 0.0, 0.0, 1.0]];

        let feature_vector:Array1<f64> = array![1.0,1.0,1.0,2.0, 0.0];
        assert_eq!(matrix_to_feature(&laplacian_matrix, 5), feature_vector);
    }


    #[test]
    fn test_matrix_to_feature_more_features(){
        let laplacian_matrix:Array2<f64> = array![[ 1.0,-0.5,-0.5,-0.5,-0.5],
                                                  [-0.5, 1.0, 0.0, 0.0, 0.0],
                                                  [-0.5, 0.0, 1.0, 0.0, 0.0],
                                                  [-0.5, 0.0, 0.0, 1.0, 0.0],
                                                  [-0.5, 0.0, 0.0, 0.0, 1.0]];

        let feature_vector:Array1<f64> = array![1.0,1.0,1.0,2.0, 0.0,0.0,0.0];
        assert_eq!(matrix_to_feature(&laplacian_matrix, 7), feature_vector);
    }


    #[test]
    fn test_matrix_to_feature_less_features(){
        let laplacian_matrix:Array2<f64> = array![[ 1.0,-0.5,-0.5,-0.5,-0.5],
                                                  [-0.5, 1.0, 0.0, 0.0, 0.0],
                                                  [-0.5, 0.0, 1.0, 0.0, 0.0],
                                                  [-0.5, 0.0, 0.0, 1.0, 0.0],
                                                  [-0.5, 0.0, 0.0, 0.0, 1.0]];

        let feature_vector:Array1<f64> = array![1.0,1.0,1.0,2.0];
        assert_eq!(matrix_to_feature(&laplacian_matrix, 4), feature_vector);
    }

    #[test]
    fn test_matrices_to_features(){
        let adj_matrix1:Array2<u64> = array![[0,1,1,1,1],
                                            [1,0,0,0,0],
                                            [1,0,0,0,0],
                                            [1,0,0,0,0],
                                            [1,0,0,0,0]];

        let adj_matrix2:Array2<u64> = array![[0,1,1],
                                             [1,0,0],
                                             [1,0,0]];

        
        let adj_matrix3:Array2<u64> = array![[0,1,1,1,1],
                                            [1,0,0,0,0],
                                            [1,0,0,0,0],
                                            [1,0,0,0,0],
                                            [1,0,0,0,0]];
       
        let features_vector:Array2<f64> = array![[1.0,1.0,1.0,2.0, 0.0],
                                                [1.0,2.0,0.0,0.0, 0.0],
                                                [1.0,1.0,1.0,2.0, 0.0]];
    
        
        let adj_matrices = vec![adj_matrix1, adj_matrix2, adj_matrix3];

        assert_eq!(adj_matrices_to_features(&adj_matrices, 5),features_vector);
    }

    #[test]
    fn test_fit_model(){
        let adj_matrix1:Array2<u64> = array![[0,1,1,1,1],
                                            [1,0,0,0,0],
                                            [1,0,0,0,0],
                                            [1,0,0,0,0],
                                            [1,0,0,0,0]];

        let adj_matrix2:Array2<u64> = array![[0,1,1],
                                             [1,0,0],
                                             [1,0,0]];

        
        let adj_matrix3:Array2<u64> = array![[0,1,1,1,1],
                                            [1,0,1,1,1],
                                            [1,1,0,1,1],
                                            [1,1,1,0,1],
                                            [1,1,1,0,1]];


        let adj_matrix4:Array2<u64> = array![[0,1,1],
                                             [1,0,1],
                                             [1,1,0]];
        

        let adj_matrices = vec![adj_matrix1, adj_matrix2, adj_matrix3, adj_matrix4];
        let target:Array1<usize> = array![0,0,1,1];
        
        let X = adj_matrices_to_features(&adj_matrices, 4);

        let dataset = linfa::Dataset::new(X, target);
        
        let model = build_model(linfa_trees::SplitQuality::Gini, None, None, None);
        let predicted_y: Array1<usize> = model.fit(&dataset).unwrap().predict(&dataset.records);
        assert_eq!(predicted_y, Array::from_iter(dataset.targets.iter().cloned()));


    }

    #[test]
    fn test_load_dataset(){
        let adj_list_path = std::path::PathBuf::from("datasets/MUTAG/MUTAG_A.txt");
        let node_graph_list_path = std::path::PathBuf::from("datasets/MUTAG/MUTAG_graph_indicator.txt");
        let graph_class_path = std::path::PathBuf::from("datasets/MUTAG/MUTAG_graph_labels.txt");


        let (dataset, tags) = load_dataset(&adj_list_path, &node_graph_list_path, &Some(graph_class_path));
        assert_eq!(tags.unwrap().len(), 188);
        assert_eq!(dataset[0].shape(), &[17,17]);
        assert_eq!(dataset.len(), 188);
    }

}
