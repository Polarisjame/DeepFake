import tensorflow as tf
import torch

def cross_replica_gather(tensor, num_replica, batch_dim=0):
  """Cross replica gather of tensors.

  Args:
    tensor: The input tensor to gather from other replica
    num_replica: The total number of replica.
    batch_dim: The batch index of the input tensor.

  Returns:
    The gathered tensor from all replica, where other tensors from other
    replica are concatenated in the batch dimension batch_dim.
  """
  ts_shape = [num_replica] + tensor.shape.as_list()
  group_assignment = [list(range(num_replica))]
  tensor = tf.raw_ops.AllToAll(
      input=tf.broadcast_to(tf.expand_dims(tensor, 0), shape=ts_shape),
      group_assignment=group_assignment,
      concat_dimension=batch_dim + 1,
      split_dimension=0,
      split_count=num_replica,
      name="AllToAllGather",
  )
  return tf.squeeze(tensor, axis=0)


a = torch.randn((8,16,512))
a_t = tf.convert_to_tensor(a.numpy())

cross_replica_gather(a_t,4)