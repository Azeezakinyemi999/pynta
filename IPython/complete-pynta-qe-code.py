import os
import yaml
import numpy as np
from ase.io import read
from ase.build import fcc111, fcc100, fcc110, bcc111, bcc100, bcc110, hcp0001
from ase.build import surface, bulk
from fireworks import LaunchPad
from pynta.main import Pynta

# Dictionary of metals and their lattice parameters (in Angstroms)
METAL_LATTICE_PARAMS = {
    'Pt': 3.92,
    'Pd': 3.89,
    'Au': 4.08,
    'Ag': 4.09,
    'Cu': 3.61,
    'Ni': 3.52,
    'Fe': 2.87,  # BCC
    'Ru': 2.71,  # HCP
    # Add more metals as needed
}

# Dictionary of oxide materials with their crystal structures
# Format: 'material': (structure, a, b, c, alpha, beta, gamma)
OXIDE_MATERIALS = {
    'Fe2O3': {  # Hematite - α-Fe2O3 (hexagonal)
        'structure': 'hematite',
        'a': 5.038,  # Å
        'b': 5.038,  # Å
        'c': 13.772,  # Å
        'alpha': 90.0,  # degrees
        'beta': 90.0,   # degrees
        'gamma': 120.0  # degrees
    },
    'Cr2O3': {  # Chromia (hexagonal, corundum structure)
        'structure': 'corundum',
        'a': 4.959,  # Å
        'b': 4.959,  # Å
        'c': 13.594,  # Å
        'alpha': 90.0,  # degrees
        'beta': 90.0,   # degrees
        'gamma': 120.0  # degrees
    }
    # Add more oxide materials as needed
}

# Dictionary mapping surface builders to their function
SURFACE_BUILDERS = {
    'fcc111': fcc111,
    'fcc100': fcc100,
    'fcc110': fcc110,
    'bcc111': bcc111,
    'bcc100': bcc100,
    'bcc110': bcc110,
    'hcp0001': hcp0001,
    # Add more surface types as needed
}

# Dictionary of Quantum ESPRESSO presets for different materials
QE_PRESETS = {
    # Metal presets
    'Pt': {
        'input_data': {
            'ecutwfc': 40.0,            # Plane-wave cutoff (Ry)
            'ecutrho': 320.0,           # Density cutoff (Ry)
            'occupations': 'smearing',
            'smearing': 'marzari-vanderbilt',
            'degauss': 0.02,            # Smearing width (Ry)
            'mixing_beta': 0.3,         # Mixing parameter
            'conv_thr': 1.0e-6,         # Convergence threshold
        },
        'pseudopotentials': {
            'Pt': 'Pt.pbe-n-rrkjus_psl.1.0.0.UPF',
        },
        'kpts': (4, 4, 1)               # k-point grid
    },
    'Fe': {
        'input_data': {
            'ecutwfc': 30.0,
            'ecutrho': 240.0,
            'occupations': 'smearing',
            'smearing': 'marzari-vanderbilt',
            'degauss': 0.02,
            'mixing_beta': 0.3,
            'conv_thr': 1.0e-6,
            'nspin': 2,                 # Spin-polarized calculation
            'starting_magnetization': {'Fe': 0.7},  # Initial magnetic moment
        },
        'pseudopotentials': {
            'Fe': 'Fe.pbe-spn-kjpaw_psl.1.0.0.UPF',
        },
        'kpts': (4, 4, 1)
    },
    # Oxide presets
    'Fe2O3': {
        'input_data': {
            'ecutwfc': 50.0,
            'ecutrho': 400.0,
            'occupations': 'smearing',
            'smearing': 'marzari-vanderbilt',
            'degauss': 0.01,
            'mixing_beta': 0.2,
            'conv_thr': 1.0e-8,
            'nspin': 2,                 # Spin-polarized calculation
            'starting_magnetization': {'Fe': 0.6, 'O': 0.0},
            # Hubbard U correction for Fe
            'lda_plus_u': True,
            'Hubbard_U': {'Fe': 4.0},
            'input_dft': 'pbe',
        },
        'pseudopotentials': {
            'Fe': 'Fe.pbe-spn-kjpaw_psl.1.0.0.UPF',
            'O': 'O.pbe-n-kjpaw_psl.1.0.0.UPF',
        },
        'kpts': (3, 3, 1)
    },
    'Cr2O3': {
        'input_data': {
            'ecutwfc': 50.0,
            'ecutrho': 400.0,
            'occupations': 'smearing',
            'smearing': 'marzari-vanderbilt',
            'degauss': 0.01,
            'mixing_beta': 0.2,
            'conv_thr': 1.0e-8,
            'nspin': 2,                 # Spin-polarized calculation
            'starting_magnetization': {'Cr': 0.6, 'O': 0.0},
            # Hubbard U correction for Cr
            'lda_plus_u': True,
            'Hubbard_U': {'Cr': 3.5},
            'input_dft': 'pbe',
        },
        'pseudopotentials': {
            'Cr': 'Cr.pbe-spn-kjpaw_psl.1.0.0.UPF',
            'O': 'O.pbe-n-kjpaw_psl.1.0.0.UPF',
        },
        'kpts': (3, 3, 1)
    }
}

def create_metal_slab(metal, surface_type, size=(3, 3, 4), vacuum=10.0, pbc=(True, True, False)):
    """
    Create a metal slab of the specified type.
    
    Args:
        metal (str): Metal symbol (e.g., 'Pt', 'Pd')
        surface_type (str): Type of surface (e.g., 'fcc111', 'bcc100')
        size (tuple): Size of the slab (x, y, z)
        vacuum (float): Vacuum space in Angstroms
        pbc (tuple): Periodic boundary conditions (x, y, z)
        
    Returns:
        ase.Atoms: Slab structure
    """
    if metal not in METAL_LATTICE_PARAMS:
        raise ValueError(f"Metal {metal} not found in lattice parameter database")
    
    if surface_type not in SURFACE_BUILDERS:
        raise ValueError(f"Surface type {surface_type} not supported")
    
    # Get the lattice parameter for the selected metal
    a = METAL_LATTICE_PARAMS[metal]
    
    # Get the appropriate surface builder function
    builder_func = SURFACE_BUILDERS[surface_type]
    
    # Create the slab
    slab = builder_func(metal, size=size, a=a, vacuum=vacuum)
    slab.pbc = pbc
    
    return slab

def create_fe110_slab(size=(3, 3, 4), vacuum=10.0, pbc=(True, True, False)):
    """
    Create a Fe(110) slab specifically.
    
    Args:
        size (tuple): Size of the slab (x, y, z)
        vacuum (float): Vacuum space in Angstroms
        pbc (tuple): Periodic boundary conditions (x, y, z)
        
    Returns:
        ase.Atoms: Slab structure
    """
    a = METAL_LATTICE_PARAMS['Fe']
    slab = bcc110('Fe', size=size, a=a, vacuum=vacuum)
    slab.pbc = pbc
    return slab

def create_oxide_slab(material, surface_indices, size=(1, 1, 1), vacuum=10.0, pbc=(True, True, False), layers=4):
    """
    Create an oxide slab of the specified material.
    
    Args:
        material (str): Oxide material name (e.g., 'Fe2O3', 'Cr2O3')
        surface_indices (tuple): Miller indices (h, k, l) for the surface
        size (tuple): Size of the slab (x, y, z)
        vacuum (float): Vacuum space in Angstroms
        pbc (tuple): Periodic boundary conditions (x, y, z)
        layers (int): Number of atomic layers in the slab
        
    Returns:
        ase.Atoms: Slab structure
    """
    if material not in OXIDE_MATERIALS:
        raise ValueError(f"Oxide material {material} not found in database")
    
    oxide_params = OXIDE_MATERIALS[material]
    
    # Create the bulk structure based on material type
    if material == 'Fe2O3':  # Hematite
        # Create a simplified hematite structure
        # Note: This is a simplified approach. For research-grade models,
        # consider importing from a .cif file or using ASE's database
        from ase import Atoms
        a = oxide_params['a']
        c = oxide_params['c']
        
        # Creating a simplified hexagonal unit cell for Fe2O3
        # (This is a simplified model - for accuracy, use a CIF file)
        bulk_structure = Atoms('Fe2O3',
                        positions=[(0, 0, 0),
                                  (a/3, 2*a/3, c/6),
                                  (a/6, a/3, c/4),
                                  (a/2, 0, c/4),
                                  (5*a/6, 2*a/3, c/4)],
                        cell=[a, a, c, 90, 90, 120],
                        pbc=True)
    
    elif material == 'Cr2O3':  # Chromia
        # Similar simplified approach for Cr2O3
        from ase import Atoms
        a = oxide_params['a']
        c = oxide_params['c']
        
        # Creating a simplified hexagonal unit cell for Cr2O3
        # (This is a simplified model - for accuracy, use a CIF file)
        bulk_structure = Atoms('Cr2O3',
                        positions=[(0, 0, 0),
                                  (a/3, 2*a/3, c/6),
                                  (a/6, a/3, c/4),
                                  (a/2, 0, c/4),
                                  (5*a/6, 2*a/3, c/4)],
                        cell=[a, a, c, 90, 90, 120],
                        pbc=True)
    
    else:
        raise NotImplementedError(f"Structure building for {material} is not implemented")
    
    # Create the surface from the bulk
    slab = surface(bulk_structure, surface_indices, layers, vacuum=vacuum)
    
    # Apply the size (supercell)
    from ase.build import make_supercell
    if size != (1, 1, 1):
        supercell_matrix = np.diag([size[0], size[1], size[2]])
        slab = make_supercell(slab, supercell_matrix)
    
    # Set the periodic boundary conditions
    slab.pbc = pbc
    
    return slab

def create_slab(material, surface_type=None, surface_indices=None, size=(3, 3, 4), 
                vacuum=10.0, pbc=(True, True, False), layers=4):
    """
    Create a slab of the specified material and surface type.
    
    Args:
        material (str): Material name (e.g., 'Pt', 'Fe2O3')
        surface_type (str): Type of surface for metals (e.g., 'fcc111')
        surface_indices (tuple): Miller indices for oxide surfaces
        size (tuple): Size of the slab
        vacuum (float): Vacuum space in Angstroms
        pbc (tuple): Periodic boundary conditions
        layers (int): Number of atomic layers for oxide slabs
        
    Returns:
        ase.Atoms: Slab structure
        str: Descriptive surface name
    """
    # Special case for Fe(110)
    if material == 'Fe' and surface_type == 'bcc110':
        return create_fe110_slab(size, vacuum, pbc), 'Fe110'
    
    # Oxide materials
    if material in OXIDE_MATERIALS:
        if surface_indices is None:
            if material == 'Fe2O3':
                surface_indices = (0, 0, 1)  # Default to (001) for Fe2O3
            elif material == 'Cr2O3':
                surface_indices = (0, 0, 1)  # Default to (001) for Cr2O3
            else:
                raise ValueError(f"Surface indices must be specified for {material}")
        
        slab = create_oxide_slab(material, surface_indices, size, vacuum, pbc, layers)
        # Create a descriptive surface name
        surface_name = f"{material}_{surface_indices[0]}{surface_indices[1]}{surface_indices[2]}"
        return slab, surface_name
    
    # Regular metals
    if material in METAL_LATTICE_PARAMS:
        if surface_type is None:
            raise ValueError(f"Surface type must be specified for metal {material}")
        
        slab = create_metal_slab(material, surface_type, size, vacuum, pbc)
        return slab, f"{material}_{surface_type.upper()}"
    
    raise ValueError(f"Unknown material: {material}")

def create_qe_kwargs(material, custom_kwargs=None):
    """
    Create Quantum ESPRESSO keyword arguments for a given material,
    combining presets with custom settings.
    
    Args:
        material (str): Material name
        custom_kwargs (dict): Custom settings to override presets
        
    Returns:
        dict: Quantum ESPRESSO settings
    """
    if material in QE_PRESETS:
        # Start with the preset
        kwargs = QE_PRESETS[material].copy()
        
        # Override with custom values if provided
        if custom_kwargs:
            # Handle input_data
            if 'input_data' in custom_kwargs:
                if 'input_data' not in kwargs:
                    kwargs['input_data'] = {}
                kwargs['input_data'].update(custom_kwargs.get('input_data', {}))
            
            # Handle pseudopotentials
            if 'pseudopotentials' in custom_kwargs:
                if 'pseudopotentials' not in kwargs:
                    kwargs['pseudopotentials'] = {}
                kwargs['pseudopotentials'].update(custom_kwargs.get('pseudopotentials', {}))
            
            # Handle k-points
            if 'kpts' in custom_kwargs:
                kwargs['kpts'] = custom_kwargs['kpts']
            
            # Handle any other keys
            for key, value in custom_kwargs.items():
                if key not in ['input_data', 'pseudopotentials', 'kpts']:
                    kwargs[key] = value
    else:
        # If no preset exists, use custom kwargs or empty dict
        kwargs = custom_kwargs if custom_kwargs else {}
    
    return kwargs

def run_pynta_workflow(
    material='Pt',
    surface_type='fcc111',
    surface_indices=None,
    size=(3, 3, 4),
    vacuum=10.0,
    pbc=(True, True, False),
    layers=4,
    base_dir=None,
    template_file="reaction_template.yaml",
    label=None,
    launchpad_file="my_launchpad.yaml",
    software="QE",  # Changed default to Quantum ESPRESSO
    software_kwargs=None,
    espresso_pseudodir=None,  # Path to pseudopotential directory
    calculate_adsorbates=True,
    calculate_transition_states=True,
    launch=True
):
    """
    Run a complete Pynta workflow for a given material and surface using Quantum ESPRESSO.
    
    Args:
        material (str): Material name (e.g., 'Pt', 'Fe2O3')
        surface_type (str): Type of surface for metals (e.g., 'fcc111')
        surface_indices (tuple): Miller indices for oxide surfaces (e.g., (0, 0, 1))
        size (tuple): Size of the slab
        vacuum (float): Vacuum space in Angstroms
        pbc (tuple): Periodic boundary conditions
        layers (int): Number of atomic layers for oxide slabs
        base_dir (str): Base directory
        template_file (str): Reaction template filename
        label (str): Label for the calculation
        launchpad_file (str): FireWorks launchpad configuration file
        software (str): Software to use for calculations ('QE' for Quantum ESPRESSO)
        software_kwargs (dict): Additional parameters for the software
        espresso_pseudodir (str): Directory containing Quantum ESPRESSO pseudopotentials
        calculate_adsorbates (bool): Whether to calculate adsorbates
        calculate_transition_states (bool): Whether to calculate transition states
        launch (bool): Whether to launch calculations immediately
    """
    # Set defaults
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if espresso_pseudodir is None:
        # Try to get from environment variable
        espresso_pseudodir = os.environ.get('ESPRESSO_PSEUDO', 
                                            '/usr/share/espresso/pseudo')
    
    # Set paths
    template_path = os.path.join(base_dir, template_file)
    
    # Create the slab
    slab, surface_name = create_slab(
        material=material,
        surface_type=surface_type,
        surface_indices=surface_indices,
        size=size,
        vacuum=vacuum,
        pbc=pbc,
        layers=layers
    )
    
    # Set label if not provided
    if label is None:
        label = f"{surface_name}_reaction"
    
    # Save the slab with informative filename
    slab_filename = f"slab_{surface_name}.xyz"
    slab_path = os.path.join(base_dir, slab_filename)
    slab.write(slab_path)
    
    # Process Quantum ESPRESSO settings
    if software.upper() == 'QE' or software.upper() == 'ESPRESSO':
        software = 'espresso'  # ASE calculator name for Quantum ESPRESSO
        
        # Create QE kwargs by combining presets with custom options
        qe_kwargs = create_qe_kwargs(material, software_kwargs)
        
        # Ensure pseudopotential directory is set
        if 'pseudodir' not in qe_kwargs and espresso_pseudodir:
            qe_kwargs['pseudodir'] = espresso_pseudodir
        
        software_kwargs = qe_kwargs
    
    # Print some debug info
    print(f"Saved slab to: {slab_path}")
    print("Material:", material)
    print("Surface:", surface_name)
    print("Software:", software)
    print("Slab cell:", slab.cell)
    print("Slab PBC:", slab.pbc)
    print("Number of atoms:", len(slab))
    
    # Load reaction templates
    with open(template_path, 'r') as f:
        templates = yaml.safe_load(f)
    
    # For oxides, we need to determine the appropriate surface_type for Pynta
    effective_surface_type = surface_type
    if material in OXIDE_MATERIALS:
        # Use a generic surface type for oxides
        effective_surface_type = "oxide"
    
    # Initialize Pynta with the slab
    pynta = Pynta(
        path=base_dir,
        rxns_file=template_path,
        surface_type=effective_surface_type,
        metal=material,  # This may need adjustment for oxides
        label=label,
        launchpad_path=os.path.join(base_dir, launchpad_file),
        slab_path=slab_path,
        software=software,
        software_kwargs=software_kwargs,
        vacuum=vacuum,
        repeats=size,
        pbc=pbc
    )
    
    # Execute the workflow
    pynta.execute(
        calculate_adsorbates=calculate_adsorbates,
        calculate_transition_states=calculate_transition_states,
        launch=launch
    )

# Example usage
if __name__ == "__main__":
    # Pt(111) with Quantum ESPRESSO
    run_pynta_workflow(
        material='Pt',
        surface_type='fcc111',
        label="Pt_111_QE",
        software="QE",
        # Custom override
        software_kwargs={
            'input_data': {
                'ecutwfc': 45.0,  # Override the preset
                'tprnfor': True,  # Print forces
                'tstress': True,  # Print stress
            },
            'kpts': (6, 6, 1),    # Finer k-point grid
        },
        espresso_pseudodir="/path/to/your/pseudos"
    )
    
    # Fe(110) with Quantum ESPRESSO
    run_pynta_workflow(
        material='Fe',
        surface_type='bcc110',
        label="Fe110_QE",
        software="QE",
        software_kwargs={
            'input_data': {
                'nspin': 2,  # Spin-polarized
                'starting_magnetization': {'Fe': 0.7}
            }
        }
    )
    
    # Fe2O3 with Quantum ESPRESSO
run_pynta_workflow(
    material='Fe2O3',
    surface_indices=(0, 0, 1),  # Same surface for comparison
    size=(2, 2, 1),
    vacuum=12.0,
    layers=3,
    label="Fe2O3_001_QE",
    software="QE",
    software_kwargs={
        'input_data': {
            'ecutwfc': 50.0,
            'ecutrho': 400.0,
            'lda_plus_u': True,
            'Hubbard_U': {'Fe': 4.0},  # Fe-specific U value
            'starting_magnetization': {'Fe': 0.6, 'O': 0.0},
            # Additional convergence settings
            'mixing_mode': 'local-TF',
            'electron_maxstep': 200,
        },
        'pseudopotentials': {
            # Use different pseudopotentials if needed
            'Fe': 'Fe.pbe-spn-kjpaw_psl.1.0.0.UPF',
            'O': 'O.pbe-n-kjpaw_psl.1.0.0.UPF',
        }
    },
    espresso_pseudodir="/path/to/your/pseudos"
)

# Cr2O3 with Quantum ESPRESSO
run_pynta_workflow(
    material='Cr2O3',
    surface_indices=(0, 0, 1),  # Same surface for comparison
    size=(2, 2, 1),
    vacuum=12.0,
    layers=3,  # Same number of layers
    label="Cr2O3_001_QE",
    software="QE",
    software_kwargs={
        'input_data': {
            'ecutwfc': 50.0,  # Same cutoffs
            'ecutrho': 400.0,
            'lda_plus_u': True,
            'Hubbard_U': {'Cr': 3.5},  # Cr-specific U value
            'starting_magnetization': {'Cr': 0.6, 'O': 0.0},
            # Additional convergence settings
            'mixing_mode': 'local-TF',
            'electron_maxstep': 200,
        },
        'pseudopotentials': {
            # Use different pseudopotentials if needed
            'Cr': 'Cr.pbe-spn-kjpaw_psl.1.0.0.UPF',
            'O': 'O.pbe-n-kjpaw_psl.1.0.0.UPF',
        }
    },
    espresso_pseudodir="/path/to/your/pseudos"
)