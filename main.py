import os
import magenta
from magenta import music
from magenta.models import melody_rnn
from magenta.music import midi_io
import tensorflow as tf
import numpy as np

# Set the environment variable for TensorFlow to use the GPU if available
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def load_dataset(midi_dir):
    """Load MIDI files from a directory."""
    midi_files = []
    for filename in os.listdir(midi_dir):
        if filename.endswith('.mid'):
            midi_file_path = os.path.join(midi_dir, filename)
            midi_files.append(midi_file_path)
    return midi_files

def train_model(midi_files):
    """Train a Melody RNN model on the provided MIDI dataset."""
    # Create a training pipeline (this is a simplified version)
    # You would typically set up a more complex training pipeline with parameters
    # and save the model after training.
    
    # Placeholder for training code
    print("Training model on MIDI files...")
    
    # Example of model training would go here
    # Use Magenta's training functions to train on the dataset
    
    print("Model training completed (this is a placeholder).")

def generate_music(model_name, output_file, total_seconds=30, temperature=1.0):
    """Generate music using a pre-trained model."""
    # Load a pre-trained Melody RNN model
    bundle = melody_rnn.sequence_generator_bundle.read_bundle_file(f'{model_name}.mag')
    generator = melody_rnn.MelodyRnnSequenceGenerator(
        model=melody_rnn.create_model(bundle), 
        steps_per_quarter=4,
        num_velocity=128,
        min_note=0,
        max_note=127
    )

    # Set the generation parameters
    num_steps = total_seconds * 4  # Assuming 4 steps per second

    # Generate a sequence
    generated_sequence = generator.generate(
        num_steps=num_steps,
        temperature=temperature,
    )

    # Save the generated sequence to a MIDI file
    midi_io.sequence_proto_to_midi_file(generated_sequence, output_file)
    print(f"Generated music saved as '{output_file}'.")

def main():
    print("Starting the AI Music Maker...")

    # Load your MIDI dataset
    midi_dir = 'path/to/your/midi/files'  # Specify the path to your MIDI files
    midi_files = load_dataset(midi_dir)

    # Train the model (optional)
    if midi_files:
        train_model(midi_files)

    # Generate music
    model_name = 'attention_rnn'  # Specify the model name
    output_file = 'generated_music.mid'  # Output MIDI file name
    total_seconds = 30  # Duration of the generated music
    temperature = 1.0  # Controls randomness of the generation

    generate_music(model_name, output_file, total_seconds, temperature)

if __name__ == "__main__":
    main()
