# compute shader
# version 430

layout( local_size_x = 1000 ) in;

uniform int dim_hd;
uniform float magnitude;
uniform float th1;
uniform float th2;

layout (std430, binding=0) buffer SSB_Picker {
  uint  clicked_x, clicked_y;
  float pick_z;
  int   pick_lock;
  readonly int   pick_id;
};

layout(std430, binding=1) buffer SSB_Layout {
  float layout_hd[];
};

layout(std430, binding=2) buffer SSB_Projection {
  float projection[];
};

layout(std430, binding=3) buffer SSB_Position {
  float positions[];
};

layout (std430, binding=4) buffer SSB_Filter {
  float score[];
};

layout(std430, binding=5) buffer SSB_Flags {
  volatile int vertexColor[];
};

layout(std430, binding=6) buffer SSB_Edgelist {
  readonly uint edges[];
};

layout(std430, binding=7) buffer SSB_isDrawn {
  volatile int isDrawn[];
};

layout(std430, binding=8) buffer SSB_Isolation {
  volatile int isIsolated[];
};

subroutine void Compute(uint);
subroutine uniform Compute compute;

subroutine (Compute)
void proj(uint invocationID) {
  float pos_x = 0, pos_y = 0;

  for (int i=0; i < dim_hd; i++) {
    pos_x += layout_hd[dim_hd * invocationID + i] * projection[2 * i];
    pos_y += layout_hd[dim_hd * invocationID + i] * projection[2 * i + 1];
  }

  positions[3 * invocationID] = pos_x * magnitude;
  positions[3 * invocationID + 1] = pos_y * magnitude;
  // positions[3 * invocationID + 2] = 0;
}

subroutine (Compute)
void find_neighbors(uint invocationID) {
  int e0 = int(edges[invocationID * 2]);
  int e1 = int(edges[invocationID * 2 + 1]);
  if (e0 == pick_id) {
    vertexColor[e1] = 2;
    positions[3 * e1 + 2] = -0.4;
  }
  if (e1 == pick_id) {
    vertexColor[e0] = 2;
    positions[3 * e0 + 2] = -0.4;
  }
}

subroutine (Compute)
void _find_neighbors(uint invocationID) {
  if (invocationID == pick_id) {
    vertexColor[invocationID] = 1;
    positions[3 * invocationID + 2] = -0.5;
  } else {
    vertexColor[invocationID] = 0;
    positions[3 * invocationID + 2] = -0.3;
  }
}

subroutine (Compute)
void clear_annotation(uint invocationID) {
  // clear vertex color flags
  vertexColor[invocationID] = 0;
}

subroutine (Compute)
void NoFilter(uint invocationID) {
  isIsolated[invocationID] = 0;
  isDrawn[invocationID] = 1;
}

subroutine (Compute)
void filtering(uint invocationID) {
  // initialize isolation
  isIsolated[invocationID] = 1;
  // Apply filtering
  float s = score[invocationID];
  if (s < th1 || th2 < s) {
    isDrawn[invocationID] = 0;
  } else {
    isDrawn[invocationID] = 1;
  }
}

subroutine (Compute)
void find_isolated(uint invocationID) {
  int e0 = int(edges[invocationID * 2]);
  int e1 = int(edges[invocationID * 2 + 1]);
  if (isDrawn[e0] == 1 && isDrawn[e1] == 1) {
    isIsolated[e0] = 0;
    isIsolated[e1] = 0;
  }
}


void main() {
  uint invocationID = gl_GlobalInvocationID.x;
  compute(invocationID);
}
