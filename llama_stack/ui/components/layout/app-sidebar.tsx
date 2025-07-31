"use client";

import {
  MessageSquareText,
  MessagesSquare,
  MoveUpRight,
  Database,
  MessageCircle,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarHeader,
} from "@/components/ui/sidebar";
// Extracted Chat Playground item
const chatPlaygroundItem = {
  title: "Chat Playground",
  url: "/chat-playground",
  icon: MessageCircle,
};

// Removed Chat Playground from log items
const logItems = [
  {
    title: "Chat Completions",
    url: "/logs/chat-completions",
    icon: MessageSquareText,
  },
  {
    title: "Responses",
    url: "/logs/responses",
    icon: MessagesSquare,
  },
  {
    title: "Vector Stores",
    url: "/logs/vector-stores",
    icon: Database,
  },
  {
    title: "Documentation",
    url: "https://llama-stack.readthedocs.io/en/latest/references/api_reference/index.html",
    icon: MoveUpRight,
  },
];

export function AppSidebar() {
  const pathname = usePathname();

  return (
    <Sidebar>
      <SidebarHeader>
        <Link href="/">Llama Stack</Link>
      </SidebarHeader>
      <SidebarContent>
        {/* Chat Playground as its own section */}
        <SidebarGroup>
          <SidebarGroupContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton
                  asChild
                  className={cn(
                    "justify-start",
                    pathname.startsWith(chatPlaygroundItem.url) &&
                      "bg-gray-200 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-900 dark:text-gray-100",
                  )}
                >
                  <Link href={chatPlaygroundItem.url}>
                    <chatPlaygroundItem.icon
                      className={cn(
                        pathname.startsWith(chatPlaygroundItem.url) && "text-gray-900 dark:text-gray-100",
                        "mr-2 h-4 w-4",
                      )}
                    />
                    <span>{chatPlaygroundItem.title}</span>
                  </Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        {/* Logs section */}
        <SidebarGroup>
          <SidebarGroupLabel>Logs</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {logItems.map((item) => {
                const isActive = pathname.startsWith(item.url);
                return (
                  <SidebarMenuItem key={item.title}>
                    <SidebarMenuButton
                      asChild
                      className={cn(
                        "justify-start",
                        isActive &&
                          "bg-gray-200 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-900 dark:text-gray-100",
                      )}
                    >
                      <Link href={item.url}>
                        <item.icon
                          className={cn(
                            isActive && "text-gray-900 dark:text-gray-100",
                            "mr-2 h-4 w-4",
                          )}
                        />
                        <span>{item.title}</span>
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                );
              })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
    </Sidebar>
  );
}
