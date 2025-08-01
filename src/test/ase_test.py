from src.ase.calculate import MatPL_calculator
from ase import io

def atoms2xyzstr(atoms):
    num_atom = atoms.get_global_number_of_atoms()
    vol = atoms.get_volume()
    pos = atoms.positions
    forces = atoms.get_forces()
    energy = atoms.get_potential_energy()
    cell = atoms.cell
    virial = -atoms.get_stress(voigt=False) * vol
    xyzstr = "%d\n" % num_atom
    xyz_head = 'Lattice="%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f" Properties=species:S:1:pos:R:3:forces:R:3 energy=%.8f'
    xyz_format = (cell[0,0],cell[0,1],cell[0,2],cell[1,0],cell[1,1],cell[1,2],cell[2,0],cell[2,1],cell[2,2],energy)
    if virial is not None:
        xyz_head += ' virial="%.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f"'
        xyz_format += (
            virial[0,0], virial[0,1], virial[0,2],
            virial[1,0], virial[1,1], virial[1,2],
            virial[2,0], virial[2,1], virial[2,2]
            )
    xyz_head += '\n'
    xyzstr += xyz_head % xyz_format
    for i in range(num_atom):
        xyzstr += "%2s %14.8f %14.8f %14.8f %14.8f %14.8f %14.8f\n" %\
        (atoms[i].symbol,pos[i,0],pos[i,1],pos[i,2],forces[i,0],forces[i,1],forces[i,2])
    return xyzstr

nep_path = "/data/home/wuxingxing/datas/debugs/ase/test_nep/nep_to_lmps.txt"
# nep_path = "/data/home/wuxingxing/datas/debugs/ase/test_nep/nep_model.ckpt"
train_path = "/data/home/wuxingxing/datas/debugs/ase/test_nep/train.xyz"
output_file = "/data/home/wuxingxing/datas/debugs/ase/test_nep/gpu_train_out_ckpt.xyz"

# nep_path = "/data/home/wuxingxing/datas/debugs/ase/test_dp/dp_model.ckpt"
# train_path = "/data/home/wuxingxing/datas/debugs/ase/test_dp/train.xyz"
# output_file = "/data/home/wuxingxing/datas/debugs/ase/test_dp/train_out_dpckpt.xyz"
# test

traj = io.read(train_path, index=":")
calc = MatPL_calculator(nep_path)
f = open(output_file, "w")
for i in range(len(traj)):
    atoms = traj[i]
    atoms.set_calculator(calc) #or  atoms.calc = calc
    f.write(atoms2xyzstr(atoms))
f.close()