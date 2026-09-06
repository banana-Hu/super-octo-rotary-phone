import type { ComposeRequest, PublishRequest, StartTaskRequest, TaskStatus, Work } from './types'

const delay = (ms = 260) => new Promise((resolve) => setTimeout(resolve, ms))
let polls = 0
const works: Work[] = Array.from({ length: 6 }, (_, index) => ({
  work_id: `demo-${index + 1}`, cover_url: '', template: index % 2 ? 'album' : 'comic',
  style: index % 3 ? 'realistic' : 'anime', nickname: `体验用户 ${index + 1}`, likes: 42 + index * 23,
}))

export const mockApi = {
  async upload(files: File[]) { await delay(); return { file_ids: files.map((_, i) => `mock-file-${i + 1}`) } },
  async startTask(_body: StartTaskRequest) { polls = 0; await delay(); return { task_id: 'mock-task-001' } },
  async getTask(taskId: string): Promise<TaskStatus> {
    await delay(); polls += 1; const done = polls >= 3
    return { task_id: taskId, status: done ? 'success' : 'processing', progress: done ? 100 : polls * 32,
      elements: done ? [1, 2, 3].map((id) => ({ id: `person-${id}`, url: '', label: `人物 ${id}` })) : [], result_url: null }
  },
  async compose(_taskId: string, _body: ComposeRequest) { await delay(500); return { result_url: '/mock-result.svg' } },
  async getWorks() { await delay(); return { items: works, next_cursor: null } },
  async publish(_body: PublishRequest) { await delay(); return { work_id: 'mock-work-001', share_url: location.href } },
}
