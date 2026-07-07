from src.ingestion.ingestion import DataIngestion
from src.config.settings import RAW_DATA_PATH
from src.validation.validation import DataValidator
from src.transformation.transform import DataTransformer
from src.loading.load import DataLoader
from src.dims_creating.dimensions import DimensionBuilder

def main():
    #ingestion
    ingestion = DataIngestion(RAW_DATA_PATH)
    df = ingestion.load_data()

    #validation
    validator = DataValidator(df)
    report = validator.generate_report()
    validator.save_report(report)
    
    print("\nValidation Report\n")
    for metric, value in report.items():
        print(f"{metric}:{value}")

    #transformation
    print("\nstarting data transformation\n")
    transformer = DataTransformer(df)
    transformer.clean_data()
    featured_df = transformer.create_features()
    transformer.save_data()

    #loading

    print("\nstarting data loading\n")
    loader = DataLoader(featured_df)
    loader.load_to_db()

    print("\ncreating dimension table\n")

    dimension_builder = DimensionBuilder()

    dimension_builder.build_dimensions()

    print("\nrunning ML customer segmentation\n")

    from src.ml.segmentation import CustomerSegmenter
    segmenter = CustomerSegmenter()
    segmenter.run_segmentation()

if __name__ == "__main__":
    main()