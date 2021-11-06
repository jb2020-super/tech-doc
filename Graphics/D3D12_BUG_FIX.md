# CREATEUNORDEREDACCESSVIEW_INVALIDDIMENSIONS

This error happened when called `IDMLBindingTable::BindOutputs`.
```
D3D12 ERROR: ID3D12Device::CreateUnorderedAccessView: The Dimensions of the View are invalid due to at least one of the following conditions. Assuming StructureByeStride 2, FirstElement (value = 0) must be between 0 and the maximum offset of the Buffer, 6220799, inclusively. With the current FirstElement, NumElements (value = 6220802) must be between 1 and 6220800, inclusively, in order that the View fit on the Buffer. [ STATE_CREATION ERROR #345: CREATEUNORDEREDACCESSVIEW_INVALIDDIMENSIONS]
```

This error means that the UAV view being created is larger than the actual size of the resource. And I solved this error by make the size of width of D3D12_RESOURCE_DESC a little larger when create ID3D12Resource.