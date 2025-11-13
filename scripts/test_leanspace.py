from itter_pipeline.leanspace_manager import LeanWorkspaceManager

if __name__ == "__main__":
    """Test LeanWorkspaceManager functionality."""
    print("\n" + "="*80)
    print("Testing LeanWorkspaceManager")
    print("="*80 + "\n")
    
    manager = LeanWorkspaceManager()

    print(manager.get_workspace_state())

    print(manager._run_lake_build())