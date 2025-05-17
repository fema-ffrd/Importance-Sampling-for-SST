from storm_generator import StormCatalogConfig, StormCatalogGenerator
import datetime

if __name__ == "__main__":
    config = StormCatalogConfig(
        geojson_path="domain.geojson",
        start_date=datetime.datetime(2011, 1, 1),
        end_date=datetime.datetime(2020, 12, 31),
        duration_hours=72,
        threshold_mm=63.5,
        min_separation_hours=120,
        max_events_per_year=10,
        output_dir="events"
    )

    generator = StormCatalogGenerator(config)
    ds = generator.open_dataset()
    event_times = generator.decluster_events(ds)
    generator.save_events(ds, event_times)
