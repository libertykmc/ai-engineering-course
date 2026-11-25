"""Проверка настоящего MCP/stdio в копии сервера с временными данными."""
import asyncio
import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

BASE = Path(__file__).parent


async def main():
    with tempfile.TemporaryDirectory(prefix='lesson11-mcp-') as temporary:
        root = Path(temporary)
        (root/'data').mkdir()
        shutil.copy(BASE/'mcp_server_basic.py', root/'mcp_server_basic.py')
        shutil.copy(BASE/'data/equipment_registry.json', root/'data/equipment_registry.json')
        (root/'data/incidents.json').write_text('[]')
        params = StdioServerParameters(command=sys.executable, args=[str(root/'mcp_server_basic.py')])
        results = []
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                assert {tool.name for tool in tools.tools} == {'get_equipment_status', 'create_incident'}
                resources = await session.list_resources()
                assert any(str(resource.uri) == 'equipment://list' for resource in resources.resources)
                resource = await session.read_resource('equipment://list')
                equipment = json.loads(resource.contents[0].text)
                assert len(equipment) == 6
                conv = next(item for item in equipment if item['id'] == 'CONV-09')
                assert conv['name'] == 'Транспортёрная лента ТЛ-9'
                assert conv['type'] == 'конвейер' and conv['location'] == 'Цех дробления'
                assert conv['status'] == 'в работе' and conv['last_service_date'] == '2026-02-10'

                async def call(name, arguments):
                    response = await session.call_tool(name, arguments)
                    result = json.loads(response.content[0].text)
                    assert isinstance(result['ok'], bool)
                    results.append((name, result))
                    print(name, json.dumps(arguments, ensure_ascii=False), '->', json.dumps(result, ensure_ascii=False), flush=True)
                    return result

                pump = await call('get_equipment_status', {'equipment_id': 'PUMP-07'})
                assert pump['ok'] and pump['status'] == 'в работе'
                assert pump == await call('get_equipment_status', {'equipment_id': 'PUMP-07'})
                assert (await call('get_equipment_status', {'equipment_id': 'CONV-09'}))['ok']
                missing = await call('get_equipment_status', {'equipment_id': 'XYZ-999'})
                assert not missing['ok'] and 'не найдено' in missing['error']
                denied = await call('create_incident', {'equipment_id': 'XYZ-999', 'description': 'Тест', 'priority': 'urgent', 'role': 'viewer'})
                assert not denied['ok'] and 'не имеет прав' in denied['error']
                missing = await call('create_incident', {'equipment_id': 'XYZ-999', 'description': 'Тест', 'priority': 'urgent', 'role': 'engineer'})
                assert not missing['ok'] and 'не найдено' in missing['error']
                invalid = await call('create_incident', {'equipment_id': 'SEP-04', 'description': 'Тест', 'priority': 'urgent', 'role': 'engineer'})
                assert not invalid['ok'] and 'приоритет' in invalid['error']
                default_role = await call('create_incident', {'equipment_id': 'SEP-04', 'description': 'Тест', 'priority': 'high'})
                assert not default_role['ok'] and 'viewer' in default_role['error']
                assert json.loads((root/'data/incidents.json').read_text()) == []
                arguments = {'equipment_id': 'SEP-04', 'description': 'Учебная вибрация', 'priority': 'high', 'role': 'engineer'}
                first = await call('create_incident', arguments)
                admin = await call('create_incident', {**arguments, 'role': 'admin', 'priority': 'low'})
                repeated = await call('create_incident', arguments)
                assert all(result['ok'] for result in (first, admin, repeated))
                saved = json.loads((root/'data/incidents.json').read_text())
                assert len(saved) == 3 and len({item['id'] for item in saved}) == 3
                for item in saved:
                    assert len(item['id']) == 8 and item['status'] == 'open'
                    assert datetime.fromisoformat(item['created_at']).utcoffset() is not None
                (root/'data/simulate_outage.flag').write_text('1')
                outage = await call('get_equipment_status', {'equipment_id': 'PUMP-07'})
                assert not outage['ok'] and 'недоступна' in outage['error']
                (root/'data/simulate_outage.flag').unlink()
                assert (await call('get_equipment_status', {'equipment_id': 'PUMP-07'}))['ok']

        audit = [json.loads(line) for line in (root/'data/audit_log.jsonl').read_text().splitlines()]
        assert len(audit) == len(results) == 13
        for entry, (name, result) in zip(audit, results):
            assert entry['action'] == name and entry['ok'] == result['ok']
            assert datetime.fromisoformat(entry['timestamp']).utcoffset() is not None
        print('\nOK: каталог и ресурс, 13 вызовов MCP, 13 записей аудита; проверки прав и приоритета, сохранение, повторная запись, сбой и восстановление.')
        print('Рабочие incidents.json и audit_log.jsonl не затрагивались: проверка использовала временную копию сервера.')


if __name__ == '__main__':
    asyncio.run(main())
