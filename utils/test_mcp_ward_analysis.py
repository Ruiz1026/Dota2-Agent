# test_mcp_ward_analysis.py
"""
测试 MCP Server 中的眼位分析功能
"""

import sys
import os

# 添加 mcp_server 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mcp_server'))

from dota2_fastmcp import analyze_match_wards, get_ward_statistics


def test_ward_statistics():
    """测试眼位统计功能"""
    print("=" * 60)
    print("测试眼位统计功能")
    print("=" * 60)
    
    # 使用一个已知的比赛ID
    match_id = 8650993496
    
    print(f"\n获取比赛 {match_id} 的眼位统计...\n")
    
    result = get_ward_statistics(match_id)
    print(result)


def test_ward_analysis():
    """测试完整的眼位分析功能"""
    print("\n" + "=" * 60)
    print("测试完整眼位分析功能")
    print("=" * 60)
    
    # 使用一个已知的比赛ID
    match_id = 8650993496
    
    print(f"\n分析比赛 {match_id} 的眼位并生成可视化...\n")
    
    result = analyze_match_wards(match_id, generate_html=True, generate_image=True)
    print(result)


if __name__ == "__main__":
    print("\n🎮 Dota 2 MCP Server - 眼位分析功能测试\n")
    
    try:
        # 测试统计功能
        test_ward_statistics()
        
        # 测试完整分析功能
        test_ward_analysis()
        
        print("\n✅ 测试完成!")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
