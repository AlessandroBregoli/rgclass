use rgclass::*;
use clap::{App, Arg, SubCommand, AppSettings};
use linfa_trees;
use linfa;
use log;
use env_logger;

fn main() {
    env_logger::init();    
    let matches = App::new("rgclass")
        .setting(AppSettings::SubcommandRequiredElseHelp)
        .author("Alessandro Bregoli")
        .about("Rust implementation of a simple graph classification algorithm (arXiv:1810.09155)")
        .arg(Arg::with_name("adj_list")
             .short("a")
             .long("adj_list")
             .value_name("Adjacency list")
             .help("Path of the adjacency list file in csv format")
             .takes_value(true)
             .required(true))
        .arg(Arg::with_name("node_graph_indicator")
             .short("n")
             .long("node_graph_indicator")
             .value_name("Node graph indicator")
             .help("Path of graph identifiers file in csv format for all nodes of all graphs")
             .takes_value(true)
             .required(true))
        .arg(Arg::with_name("graph_labels")
             .short("g")
             .long("graph_labels")
             .value_name("Graph label")
             .help("Path of the class labels for all graphs in the dataset in csv format")
             .takes_value(true)
             .required(true))
        .subcommand(SubCommand::with_name("cross_validate")
                    .about("Execute cross validation with a decision tree classifier")
                    .arg(Arg::with_name("k_fold")
                         .short("k")
                         .long("k_fold")
                         .value_name("K fold")
                         .help("Number of folds to apply")
                         .takes_value(true)
                         .required(true))
                    .arg(Arg::with_name("n_features")
                         .short("f")
                         .long("n_features")
                         .value_name("Number of features")
                         .help("Number of features to extract from each graph")
                         .takes_value(true)
                         .required(true))
                    .arg(Arg::with_name("entropy")
                         .short("e")
                         .long("entropy")
                         .value_name("Entropy")
                         .help("If presents, the model uses the entropy as split metric. Otherwise, it uses gini")
                         .takes_value(false)
                         .required(false))
                    .arg(Arg::with_name("max_depth")
                         .short("d")
                         .long("max_depth")
                         .value_name("Max depth")
                         .help("Set the optional limit to the depth of the decision tree")
                         .takes_value(true)
                         .required(false))
                    .arg(Arg::with_name("min_weight_split")
                         .short("s")
                         .long("min_weight_split")
                         .value_name("Min weight split")
                         .help("Set the minimum number of samples required to split a node.")
                         .takes_value(true)
                         .required(false))
                    .arg(Arg::with_name("min_weight_leaf")
                         .short("l")
                         .long("min_weight_leaf")
                         .value_name("Min weight split")
                         .help("Set the minimum number of samples that a split has to place in each leaf.")
                         .takes_value(true)
                         .required(false))).get_matches();
    log::info!("Start parsing parameters");
    let adg_list_path = std::path::PathBuf::from(matches.value_of("adj_list").unwrap());
    let node_graph_list_path = std::path::PathBuf::from(matches.value_of("node_graph_indicator").unwrap());
    let graph_class_path = std::path::PathBuf::from(matches.value_of("graph_labels").unwrap());

    log::info!("Load dataset");
    let (adj_matricres, y) = load_dataset(&adg_list_path, &node_graph_list_path, &Some(graph_class_path));


    if let Some(cross_validate) = matches.subcommand_matches("cross_validate") {
        let split_quality = if cross_validate.is_present("entropy") {
            linfa_trees::SplitQuality::Entropy
        } else {
            linfa_trees::SplitQuality::Gini
        };

        let max_depth = match cross_validate.value_of("max_depth") {
            Some(x) => Some(x.parse::<usize>().unwrap()),
            None => None
        };

        let min_weight_split = match cross_validate.value_of("min_weight_split") {
            Some(x) => Some(x.parse::<f32>().unwrap()),
            None => None
        };

        let min_weight_leaf = match cross_validate.value_of("min_weight_leaf") {
            Some(x) => Some(x.parse::<f32>().unwrap()),
            None => None
        };
        
        let k_fold: usize = cross_validate.value_of("k_fold").unwrap().parse().unwrap();
        let n_features: usize = cross_validate.value_of("n_features").unwrap().parse().unwrap();
        
        log::info!("Build model");
        let model = build_model(split_quality, max_depth, min_weight_split, min_weight_leaf);
        log::info!("Computing features");
        let X = adj_matrices_to_features(&adj_matricres, n_features);
        log::info!("Start cross validation");
        let accuracy = compute_cross_validation(model, k_fold, linfa::Dataset::new(X, y.unwrap()));
        println!("Accuracy: {}", accuracy);



    }




}


