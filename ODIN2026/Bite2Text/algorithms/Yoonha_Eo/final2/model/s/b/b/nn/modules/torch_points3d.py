import logging
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


# Define PLY types
ply_dtypes = dict(
    [
        (b"int8", "i1"),
        (b"char", "i1"),
        (b"uint8", "u1"),
        (b"uchar", "u1"),
        (b"int16", "i2"),
        (b"short", "i2"),
        (b"uint16", "u2"),
        (b"ushort", "u2"),
        (b"int32", "i4"),
        (b"int", "i4"),
        (b"uint32", "u4"),
        (b"uint", "u4"),
        (b"float32", "f4"),
        (b"float", "f4"),
        (b"float64", "f8"),
        (b"double", "f8"),
    ]
)

# Numpy reader format
valid_formats = {"ascii": "", "binary_big_endian": ">", "binary_little_endian": "<"}

def parse_header(plyfile, ext):
    # Variables
    line = []
    properties = []
    num_points = None

    while b"end_header" not in line and line != b"":
        line = plyfile.readline()

        if b"element" in line:
            line = line.split()
            num_points = int(line[2])

        elif b"property" in line:
            line = line.split()
            properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))

    return num_points, properties

def parse_mesh_header(plyfile, ext):
    # Variables
    line = []
    vertex_properties = []
    num_points = None
    num_faces = None
    current_element = None

    while b"end_header" not in line and line != b"":
        line = plyfile.readline()

        # Find point element
        if b"element vertex" in line:
            current_element = "vertex"
            line = line.split()
            num_points = int(line[2])

        elif b"element face" in line:
            current_element = "face"
            line = line.split()
            num_faces = int(line[2])

        elif b"property" in line:
            if current_element == "vertex":
                line = line.split()
                vertex_properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))
            elif current_element == "vertex":
                if not line.startswith("property list uchar int"):
                    raise ValueError("Unsupported faces property : " + line)

    return num_points, num_faces, vertex_properties


def read_ply(filename, triangular_mesh=False):
    """
    Read ".ply" files
    Parameters
    ----------
    filename : string
        the name of the file to read.
    Returns
    -------
    result : array
        data stored in the file
    Examples
    --------
    Store data in file
    >>> points = np.random.rand(5, 3)
    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example.ply', [points, values], ['x', 'y', 'z', 'values'])
    Read the file
    >>> data = read_ply('example.ply')
    >>> values = data['values']
    array([0, 0, 1, 1, 0])

    >>> points = np.vstack((data['x'], data['y'], data['z'])).T
    array([[ 0.466  0.595  0.324]
           [ 0.538  0.407  0.654]
           [ 0.850  0.018  0.988]
           [ 0.395  0.394  0.363]
           [ 0.873  0.996  0.092]])
    """

    with open(filename, "rb") as plyfile:

        # Check if the file start with ply
        if b"ply" not in plyfile.readline():
            raise ValueError("The file does not start whith the word ply")

        # get binary_little/big or ascii
        fmt = plyfile.readline().split()[1].decode()
        if fmt == "ascii":
            raise ValueError("The file is not binary")

        # get extension for building the numpy dtypes
        ext = valid_formats[fmt]

        # PointCloud reader vs mesh reader
        if triangular_mesh:

            # Parse header
            num_points, num_faces, properties = parse_mesh_header(plyfile, ext)

            # Get point data
            vertex_data = np.fromfile(plyfile, dtype=properties, count=num_points)

            # Get face data
            face_properties = [
                ("k", ext + "u1"),
                ("v1", ext + "i4"),
                ("v2", ext + "i4"),
                ("v3", ext + "i4"),
            ]
            faces_data = np.fromfile(plyfile, dtype=face_properties, count=num_faces)

            # Return vertex data and concatenated faces
            faces = np.vstack((faces_data["v1"], faces_data["v2"], faces_data["v3"])).T
            data = [vertex_data, faces]

        else:

            # Parse header
            num_points, properties = parse_header(plyfile, ext)

            # Get data
            data = np.fromfile(plyfile, dtype=properties, count=num_points)

    return data


def header_properties(field_list, field_names):

    # List of lines to write
    lines = []

    # First line describing element vertex
    lines.append("element vertex %d" % field_list[0].shape[0])

    # Properties lines
    i = 0
    for fields in field_list:
        for field in fields.T:
            lines.append("property %s %s" % (field.dtype.name, field_names[i]))
            i += 1

    return lines


def write_ply(filename, field_list, field_names, triangular_faces=None):
    """
    Write ".ply" files
    Parameters
    ----------
    filename : Path
        the path of the file to which the data is saved. A '.ply' extension will be appended to the
        file name if it does no already have one.
    field_list : list, tuple, numpy array
        the fields to be saved in the ply file. Either a numpy array, a list of numpy arrays or a
        tuple of numpy arrays. Each 1D numpy array and each column of 2D numpy arrays are considered
        as one field.
    field_names : list
        the name of each fields as a list of strings. Has to be the same length as the number of
        fields.
    Examples
    --------
    >>> points = np.random.rand(10, 3)
    >>> write_ply('example1.ply', points, ['x', 'y', 'z'])
    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example2.ply', [points, values], ['x', 'y', 'z', 'values'])
    >>> colors = np.random.randint(255, size=(10,3), dtype=np.uint8)
    >>> field_names = ['x', 'y', 'z', 'red', 'green', 'blue', values']
    >>> write_ply('example3.ply', [points, colors, values], field_names)
    """

    # Format list input to the right form
    field_list = list(field_list) if (type(field_list) == list or type(field_list) == tuple) else list((field_list,))
    for i, field in enumerate(field_list):
        if field.ndim < 2:
            field_list[i] = field.reshape(-1, 1)
        if field.ndim > 2:
            log.info("fields have more than 2 dimensions")
            return False

    # check all fields have the same number of data
    n_points = [field.shape[0] for field in field_list]
    if not np.all(np.equal(n_points, n_points[0])):
        log.info("wrong field dimensions")
        return False

    # Check if field_names and field_list have same nb of column
    n_fields = np.sum([field.shape[1] for field in field_list])
    if n_fields != len(field_names):
        log.info("wrong number of field names")
        return False

    # Add extension if not there
    if not filename.name.endswith(".ply"):
        filename += ".ply"

    # open in text mode to write the header
    with open(filename, "w") as plyfile:

        # First magical word
        header = ["ply"]

        # Encoding format
        header.append("format binary_" + sys.byteorder + "_endian 1.0")

        # Points properties description
        header.extend(header_properties(field_list, field_names))

        # Add faces if needded
        if triangular_faces is not None:
            header.append("element face {:d}".format(triangular_faces.shape[0]))
            header.append("property list uchar int vertex_indices")

        # End of header
        header.append("end_header")

        # Write all lines
        for line in header:
            plyfile.write("%s\n" % line)

    # open in binary/append to use tofile
    with open(filename, "ab") as plyfile:

        # Create a structured array
        i = 0
        type_list = []
        for fields in field_list:
            for field in fields.T:
                type_list += [(field_names[i], field.dtype.str)]
                i += 1
        data = np.empty(field_list[0].shape[0], dtype=type_list)
        i = 0
        for fields in field_list:
            for field in fields.T:
                data[field_names[i]] = field
                i += 1

        data.tofile(plyfile)

        if triangular_faces is not None:
            triangular_faces = triangular_faces.astype(np.int32)
            type_list = [("k", "uint8")] + [(str(ind), "int32") for ind in range(3)]
            data = np.empty(triangular_faces.shape[0], dtype=type_list)
            data["k"] = np.full((triangular_faces.shape[0],), 3, dtype=np.uint8)
            data["0"] = triangular_faces[:, 0]
            data["1"] = triangular_faces[:, 1]
            data["2"] = triangular_faces[:, 2]
            data.tofile(plyfile)

    return True


def kernel_point_optimization_debug(
    radius, num_points, num_kernels=1, dimension=3, fixed="center", ratio=1.0, verbose=0
):
    """
    Creation of kernel point via optimization of potentials.
    :param radius: Radius of the kernels
    :param num_points: points composing kernels
    :param num_kernels: number of wanted kernels
    :param dimension: dimension of the space
    :param fixed: fix position of certain kernel points ('none', 'center' or 'verticals')
    :param ratio: ratio of the radius where you want the kernels points to be placed
    :param verbose: display option
    :return: points [num_kernels, num_points, dimension]
    """

    #######################
    # Parameters definition
    #######################

    # Radius used for optimization (points are rescaled afterwards)
    radius0 = 1
    diameter0 = 2

    # Factor multiplicating gradients for moving points (~learning rate)
    moving_factor = 1e-2
    continuous_moving_decay = 0.9995

    # Gradient threshold to stop optimization
    thresh = 1e-5

    # Gradient clipping value
    clip = 0.05 * radius0

    #######################
    # Kernel initialization
    #######################

    # Random kernel points
    kernel_points = np.random.rand(num_kernels * num_points - 1, dimension) * diameter0 - radius0
    while kernel_points.shape[0] < num_kernels * num_points:
        new_points = np.random.rand(num_kernels * num_points - 1, dimension) * diameter0 - radius0
        kernel_points = np.vstack((kernel_points, new_points))
        d2 = np.sum(np.power(kernel_points, 2), axis=1)
        kernel_points = kernel_points[d2 < 0.5 * radius0 * radius0, :]
    kernel_points = kernel_points[: num_kernels * num_points, :].reshape((num_kernels, num_points, -1))

    # Optionnal fixing
    if fixed == "center":
        kernel_points[:, 0, :] *= 0
    if fixed == "verticals":
        kernel_points[:, :3, :] *= 0
        kernel_points[:, 1, -1] += 2 * radius0 / 3
        kernel_points[:, 2, -1] -= 2 * radius0 / 3

    #####################
    # Kernel optimization
    #####################

    # Initiate figure
    if verbose > 1:
        import matplotlib.pyplot as plt
        fig = plt.figure()

    saved_gradient_norms = np.zeros((10000, num_kernels))
    old_gradient_norms = np.zeros((num_kernels, num_points))
    for iter in range(10000):

        # Compute gradients
        # *****************

        # Derivative of the sum of potentials of all points
        A = np.expand_dims(kernel_points, axis=2)
        B = np.expand_dims(kernel_points, axis=1)
        interd2 = np.sum(np.power(A - B, 2), axis=-1)
        inter_grads = (A - B) / (np.power(np.expand_dims(interd2, -1), 3 / 2) + 1e-6)
        inter_grads = np.sum(inter_grads, axis=1)

        # Derivative of the radius potential
        circle_grads = 10 * kernel_points

        # All gradients
        gradients = inter_grads + circle_grads

        if fixed == "verticals":
            gradients[:, 1:3, :-1] = 0

        # Stop condition
        # **************

        # Compute norm of gradients
        gradients_norms = np.sqrt(np.sum(np.power(gradients, 2), axis=-1))
        saved_gradient_norms[iter, :] = np.max(gradients_norms, axis=1)

        # Stop if all moving points are gradients fixed (low gradients diff)

        if fixed == "center" and np.max(np.abs(old_gradient_norms[:, 1:] - gradients_norms[:, 1:])) < thresh:
            break
        elif fixed == "verticals" and np.max(np.abs(old_gradient_norms[:, 3:] - gradients_norms[:, 3:])) < thresh:
            break
        elif np.max(np.abs(old_gradient_norms - gradients_norms)) < thresh:
            break
        old_gradient_norms = gradients_norms

        # Move points
        # ***********

        # Clip gradient to get moving dists
        moving_dists = np.minimum(moving_factor * gradients_norms, clip)

        # Fix central point
        if fixed == "center":
            moving_dists[:, 0] = 0
        if fixed == "verticals":
            moving_dists[:, 0] = 0

        # Move points
        kernel_points -= np.expand_dims(moving_dists, -1) * gradients / np.expand_dims(gradients_norms + 1e-6, -1)

        if verbose:
            log.info("iter {:5d} / max grad = {:f}".format(iter, np.max(gradients_norms[:, 3:])))
        if verbose > 1:
            import matplotlib.pyplot as plt
            plt.clf()
            plt.plot(kernel_points[0, :, 0], kernel_points[0, :, 1], ".")
            circle = plt.Circle((0, 0), radius, color="r", fill=False)
            fig.axes[0].add_artist(circle)
            fig.axes[0].set_xlim((-radius * 1.1, radius * 1.1))
            fig.axes[0].set_ylim((-radius * 1.1, radius * 1.1))
            fig.axes[0].set_aspect("equal")
            plt.draw()
            plt.pause(0.001)
            plt.show(block=False)
            log.info(moving_factor)

        # moving factor decay
        moving_factor *= continuous_moving_decay

    # Rescale radius to fit the wanted ratio of radius
    r = np.sqrt(np.sum(np.power(kernel_points, 2), axis=-1))
    kernel_points *= ratio / np.mean(r[:, 1:])

    # Rescale kernels with real radius
    return kernel_points * radius, saved_gradient_norms


def load_kernels(radius, num_kpoints, num_kernels, dimension, fixed):

    # Number of tries in the optimization process, to ensure we get the most stable disposition
    num_tries = 100

    # Kernel directory
    kernel_dir = Path(__file__).parent / 'kernels/dispositions'
    if not kernel_dir.exists():
        kernel_dir.mkdir(parents=True)

    # Kernel_file
    if dimension == 3:
        kernel_file = kernel_dir / f'k_{num_kpoints:03d}_{fixed:s}.ply'
    elif dimension == 2:
        kernel_file = kernel_dir / f'k_{num_kpoints:03d}_{fixed:s}_2D.ply'
    else:
        raise ValueError("Unsupported dimpension of kernel : " + str(dimension))

    # Check if already done
    if not kernel_file.exists():

        # Create kernels
        kernel_points, grad_norms = kernel_point_optimization_debug(
            1.0,
            num_kpoints,
            num_kernels=num_tries,
            dimension=dimension,
            fixed=fixed,
            verbose=0,
        )

        # Find best candidate
        best_k = np.argmin(grad_norms[-1, :])

        # Save points
        original_kernel = kernel_points[best_k, :, :]
        write_ply(kernel_file, original_kernel, ["x", "y", "z"])

    else:
        data = read_ply(kernel_file)
        original_kernel = np.vstack((data["x"], data["y"], data["z"])).T

    # N.B. 2D kernels are not supported yet
    if dimension == 2:
        return original_kernel

    # Random rotations depending of the fixed points
    if fixed == "verticals":

        # Create random rotations
        thetas = np.random.rand(num_kernels) * 2 * np.pi
        c, s = np.cos(thetas), np.sin(thetas)
        R = np.zeros((num_kernels, 3, 3), dtype=np.float32)
        R[:, 0, 0] = c
        R[:, 1, 1] = c
        R[:, 2, 2] = 1
        R[:, 0, 1] = s
        R[:, 1, 0] = -s

        # Scale kernels
        original_kernel = radius * np.expand_dims(original_kernel, 0)

        # Rotate kernels
        kernels = np.matmul(original_kernel, R)

    else:

        # Create random rotations
        u = np.ones((num_kernels, 3))
        v = np.ones((num_kernels, 3))
        wrongs = np.abs(np.sum(u * v, axis=1)) > 0.99
        while np.any(wrongs):
            new_u = np.random.rand(num_kernels, 3) * 2 - 1
            new_u = new_u / np.expand_dims(np.linalg.norm(new_u, axis=1) + 1e-9, -1)
            u[wrongs, :] = new_u[wrongs, :]
            new_v = np.random.rand(num_kernels, 3) * 2 - 1
            new_v = new_v / np.expand_dims(np.linalg.norm(new_v, axis=1) + 1e-9, -1)
            v[wrongs, :] = new_v[wrongs, :]
            wrongs = np.abs(np.sum(u * v, axis=1)) > 0.99

        # Make v perpendicular to u
        v -= np.expand_dims(np.sum(u * v, axis=1), -1) * u
        v = v / np.expand_dims(np.linalg.norm(v, axis=1) + 1e-9, -1)

        # Last rotation vector
        w = np.cross(u, v)
        R = np.stack((u, v, w), axis=-1)

        # Scale kernels
        original_kernel = radius * np.expand_dims(original_kernel, 0)

        # Rotate kernels
        kernels = np.matmul(original_kernel, R)

        # Add a small noise
        kernels = kernels
        kernels = kernels + np.random.normal(scale=radius * 0.01, size=kernels.shape)

    return kernels


def add_ones(query_points, x, add_one):
    if add_one:
        ones = torch.ones(query_points.shape[0], dtype=torch.float).unsqueeze(-1).to(query_points.device)
        if x is not None:
            x = torch.cat([ones.to(x.dtype), x], dim=-1)
        else:
            x = ones
    return x


def radius_gaussian(sq_r, sig, eps=1e-9):
    """
    Compute a radius gaussian (gaussian of distance)
    :param sq_r: input radiuses [dn, ..., d1, d0]
    :param sig: extents of gaussians [d1, d0] or [d0] or float
    :return: gaussian of sq_r [dn, ..., d1, d0]
    """
    return torch.exp(-sq_r / (2 * sig**2 + eps))


def gather(x, idx, method=2):
    """
    https://github.com/pytorch/pytorch/issues/15245
    implementation of a custom gather operation for faster backwards.
    :param x: input with shape [N, D_1, ... D_d]
    :param idx: indexing with shape [n_1, ..., n_m]
    :param method: Choice of the method
    :return: x[idx] with shape [n_1, ..., n_m, D_1, ... D_d]
    """
    idx[idx == -1] = x.shape[0] - 1  # Shadow point
    if method == 0:
        return x[idx]
    elif method == 1:
        x = x.unsqueeze(1)
        x = x.expand((-1, idx.shape[-1], -1))
        idx = idx.unsqueeze(2)
        idx = idx.expand((-1, -1, x.shape[-1]))
        return x.gather(0, idx)
    elif method == 2:
        for i, ni in enumerate(idx.size()[1:]):
            x = x.unsqueeze(i + 1)
            new_s = list(x.size())
            new_s[i + 1] = ni
            x = x.expand(new_s)
        n = len(idx.size())
        for i, di in enumerate(x.size()[n:]):
            idx = idx.unsqueeze(i + n)
            new_s = list(idx.size())
            new_s[i + n] = di
            idx = idx.expand(new_s)
        return x.gather(0, idx)
    else:
        raise ValueError("Unkown method")


def KPConv_ops(
    query_points,
    support_points,
    neighbors_indices,
    features,
    K_points,
    K_values,
    KP_extent,
    KP_influence,
    aggregation_mode,
):
    """
    This function creates a graph of operations to define Kernel Point Convolution in tensorflow. See KPConv function
    above for a description of each parameter
    :param query_points: float32[n_points, dim] - input query points (center of neighborhoods)
    :param support_points: float32[n0_points, dim] - input support points (from which neighbors are taken)
    :param neighbors_indices: int32[n_points, n_neighbors] - indices of neighbors of each point
    :param features: float32[n0_points, in_fdim] - input features
    :param K_values: float32[n_kpoints, in_fdim, out_fdim] - weights of the kernel
    :param fixed: string in ('none', 'center' or 'verticals') - fix position of certain kernel points
    :param KP_extent: float32 - influence radius of each kernel point
    :param KP_influence: string in ('constant', 'linear', 'gaussian') - influence function of the kernel points
    :param aggregation_mode: string in ('closest', 'sum') - whether to sum influences, or only keep the closest
    :return:                    [n_points, out_fdim]
    """

    # Get variables
    int(K_points.shape[0])

    # Add a fake point in the last row for shadow neighbors
    shadow_point = torch.ones_like(support_points[:1, :]) * 1e6
    support_points = torch.cat([support_points, shadow_point], dim=0)

    # Get neighbor points [n_points, n_neighbors, dim]
    neighbors = gather(support_points, neighbors_indices)

    # Center every neighborhood
    neighbors = neighbors - query_points.unsqueeze(1)

    # Get all difference matrices [n_points, n_neighbors, n_kpoints, dim]
    neighbors.unsqueeze_(2)
    differences = neighbors - K_points

    # Get the square distances [n_points, n_neighbors, n_kpoints]
    sq_distances = torch.sum(differences**2, dim=3)

    # Get Kernel point influences [n_points, n_kpoints, n_neighbors]
    if KP_influence == "constant":
        # Every point get an influence of 1.
        all_weights = torch.ones_like(sq_distances)
        all_weights = all_weights.transpose(2, 1)

    elif KP_influence == "linear":
        # Influence decrease linearly with the distance, and get to zero when d = KP_extent.
        all_weights = torch.clamp(1 - torch.sqrt(sq_distances) / KP_extent, min=0.0)
        all_weights = all_weights.transpose(2, 1)

    elif KP_influence == "gaussian":
        # Influence in gaussian of the distance.
        sigma = KP_extent * 0.3
        all_weights = radius_gaussian(sq_distances, sigma)
        all_weights = all_weights.transpose(2, 1)
    else:
        raise ValueError("Unknown influence function type (config.KP_influence)")

    # In case of closest mode, only the closest KP can influence each point
    if aggregation_mode == "closest":
        neighbors_1nn = torch.argmin(sq_distances, dim=-1)
        all_weights *= torch.transpose(torch.nn.functional.one_hot(neighbors_1nn, K_points.shape[0]), 1, 2)

    elif aggregation_mode != "sum":
        raise ValueError("Unknown convolution mode. Should be 'closest' or 'sum'")

    features = torch.cat([features, torch.zeros_like(features[:1, :])], dim=0)

    # Get the features of each neighborhood [n_points, n_neighbors, in_fdim]
    neighborhood_features = gather(features, neighbors_indices)

    # Apply distance weights [n_points, n_kpoints, in_fdim]
    weighted_features = torch.matmul(all_weights, neighborhood_features)

    # Apply network weights [n_kpoints, n_points, out_fdim]
    weighted_features = weighted_features.permute(1, 0, 2)
    kernel_outputs = torch.matmul(weighted_features, K_values)

    # Convolution sum to get [n_points, out_fdim]
    output_features = torch.sum(kernel_outputs, dim=0)

    return output_features


class KPConvLayer(torch.nn.Module):
    """
    apply the kernel point convolution on a point cloud
    NB : it is the original version of KPConv, it is not the message passing version
    attributes:
    num_inputs : dimension of the input feature
    num_outputs : dimension of the output feature
    point_influence: influence distance of a single point (sigma * grid_size)
    n_kernel_points=15
    fixed="center"
    KP_influence="linear"
    aggregation_mode="sum"
    dimension=3
    """

    _INFLUENCE_TO_RADIUS = 1.5

    def __init__(
        self,
        num_inputs,
        num_outputs,
        point_influence,
        n_kernel_points=15,
        fixed="center",
        KP_influence="linear",
        aggregation_mode="sum",
        dimension=3,
        add_one=False,
        **kwargs
    ):
        super(KPConvLayer, self).__init__()
        self.kernel_radius = self._INFLUENCE_TO_RADIUS * point_influence
        self.point_influence = point_influence
        self.add_one = add_one
        self.num_inputs = num_inputs + self.add_one * 1
        self.num_outputs = num_outputs

        self.KP_influence = KP_influence
        self.n_kernel_points = n_kernel_points
        self.aggregation_mode = aggregation_mode

        # Initial kernel extent for this layer
        K_points_numpy = load_kernels(
            self.kernel_radius,
            n_kernel_points,
            num_kernels=1,
            dimension=dimension,
            fixed=fixed,
        )

        self.K_points = nn.Parameter(
            torch.from_numpy(K_points_numpy.reshape((n_kernel_points, dimension))).to(torch.float),
            requires_grad=False,
        )

        weights = torch.empty([n_kernel_points, self.num_inputs, num_outputs], dtype=torch.float)
        torch.nn.init.xavier_normal_(weights)
        self.weight = nn.Parameter(weights)

    def forward(self, query_points, support_points, neighbors, x):
        """
        - query_points(torch Tensor): query of size N x 3
        - support_points(torch Tensor): support points of size N0 x 3
        - neighbors(torch Tensor): neighbors of size N x M
        - features : feature of size N0 x d (d is the number of inputs)
        """
        x = add_ones(support_points, x, self.add_one)

        new_feat = KPConv_ops(
            query_points,
            support_points,
            neighbors,
            x,
            self.K_points,
            self.weight,
            self.point_influence,
            self.KP_influence,
            self.aggregation_mode,
        )
        return new_feat

    def __repr__(self):
        return "KPConvLayer(InF: %i, OutF: %i, kernel_pts: %i, radius: %.2f, KP_influence: %s, Add_one: %s)" % (
            self.num_inputs,
            self.num_outputs,
            self.n_kernel_points,
            self.kernel_radius,
            self.KP_influence,
            self.add_one,
        )
    

class BaseModule(nn.Module):
    """Base module class with some basic additions to the pytorch Module class"""

    @property
    def nb_params(self):
        """This property is used to return the number of trainable parameters for a given layer
        It is useful for debugging and reproducibility.
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        self._nb_params = sum([np.prod(p.size()) for p in model_parameters])
        return self._nb_params
    

class FastBatchNorm1d(BaseModule):
    def __init__(self, num_features, momentum=0.1, **kwargs):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(num_features, momentum=momentum, **kwargs)

    def _forward_dense(self, x):
        return self.batch_norm(x.permute(0, 2, 1)).permute(0, 2, 1)

    def _forward_sparse(self, x):
        """Batch norm 1D is not optimised for 2D tensors. The first dimension is supposed to be
        the batch and therefore not very large. So we introduce a custom version that leverages BatchNorm1D
        in a more optimised way
        """
        x = x.unsqueeze(2)
        x = x.transpose(0, 2)
        x = self.batch_norm(x)
        x = x.transpose(0, 2)
        return x.squeeze(dim=2)

    def forward(self, x):
        if x.dim() == 2:
            return self._forward_sparse(x)
        elif x.dim() == 3:
            return self._forward_dense(x)
        else:
            raise ValueError("Non supported number of dimensions {}".format(x.dim()))
