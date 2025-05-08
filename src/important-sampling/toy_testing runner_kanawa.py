from toy_testing import WatershedModel, StormModel, PlotView

def main():
    try:
        # Define constants
        transposition_domain_size = (100, 100)

        # Generate storm templates
        print("Generating storm templates...")
        storm_templates = StormModel.generate_templates(50, 5, 50, mean=30, std=3, noise_level=0.1)

        # Generate watershed cells
        print("Generating watershed cells...")
        watershed_cells = WatershedModel.generate_random_cells(transposition_domain_size, 225, aspect=1 / 20)

        # Plot domain
        print("Creating and saving domain plot...")
        plot = PlotView.plot_domain(transposition_domain_size, watershed_cells)
        PlotView.save_plot(plot, "domain_plot.png")

        # Run simulation (placeholder for future logic)
        print("Simulation logic can be added here.")
    except Exception as e:
        print(f"An error occurred in the main pipeline: {e}")

if __name__ == "__main__":
    main()